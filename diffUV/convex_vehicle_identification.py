"""Bounded-size MOSEK formulation of the notebook's constrained least squares.

No prior, no parameter feedback, no alternate optimizer or posterior sampling.
QR compression removes only an additive constant from the original objective.
"""
from dataclasses import dataclass, replace
from time import perf_counter
import cvxpy as cp
import numpy as np
from .system_id import MarineVehicleEstimator


@dataclass(frozen=True)
class ConvexVehicleFit:
    parameters: np.ndarray
    residual_rms: np.ndarray
    solver_seconds: float
    wall_seconds: float
    equality_reduced_rank: int
    robust_iterations: int = 0
    robust_relative_objective_change: float = 0.


class ConvexVehicleIdentifier:
    """Same feasible set as the notebook, with explicit deployment configuration.

    Rank is diagnostic only: a successful solve is NOT permission to update
    control, nor a claim of identifiable/accurate individual parameters.
    Call on a worker thread, not the real-time control thread.
    """
    def __init__(self, *, weight_bounds, solver_budget=.5, inertia_floor=1e-6):
        bounds=np.asarray(weight_bounds,float)
        if (bounds.shape!=(2,) or not np.isfinite(bounds).all()
                or bounds[0]<=0 or bounds[0]>bounds[1]
                or not np.isfinite(solver_budget) or solver_budget<=0
                or not np.isfinite(inertia_floor) or inertia_floor<=0):
            raise ValueError('Invalid physical bounds or solver budget')
        self.bounds,self.budget,self.floor=bounds,float(solver_budget),float(inertia_floor)
        self.coordinates=cp.Variable(27)
        self.parameter_scale=cp.Parameter(27,nonneg=True,value=np.ones(27))
        self.theta=cp.multiply(self.parameter_scale,self.coordinates)
        self.factor=cp.Parameter((27,27))
        self.target=cp.Parameter(27)
        M=MarineVehicleEstimator.M_from_uv(self,self.theta)
        self.problem=cp.Problem(cp.Minimize(cp.sum_squares(self.factor@self.coordinates-self.target)),
            [M >> self.floor*np.eye(6),M==M.T,self.theta[15:]<=0,
             self.theta[10]>=bounds[0],self.theta[10]<=bounds[1]])
        count=3 if bounds[0]==bounds[1] else 2
        E=np.zeros((count,27));E[0,3]=E[1,4]=1;E[0,6]=E[1,5]=-1
        if count==3:
            E[2,10]=1
        self.basis=np.linalg.svd(E,full_matrices=True)[2][count:].T

    def fit(self, design, wrench):
        started=perf_counter()
        A=np.asarray(design,float);b=np.asarray(wrench,float)
        if (A.ndim!=2 or A.shape[1]!=27 or len(A)<27 or len(A)%6
                or b.shape!=(len(A),) or not np.isfinite(A).all() or not np.isfinite(b).all()):
            raise ValueError('Expected >=5 complete six-axis finite regressor samples')
        norms=np.linalg.norm(A,axis=0)
        scales=np.ones(27)
        np.divide(1.,norms,out=scales,where=norms>0)
        # Invertible coordinate preconditioning only, not bounds or a prior.
        # Limit extreme coordinate units when a column is nearly unexcited.
        scales=np.clip(scales,1e-3,1e3)
        self.parameter_scale.value=scales
        Q,R=np.linalg.qr(A*scales,mode='reduced')
        # Keep physical residual units. Dividing by ||R|| makes tiny, but
        # informative residual changes in weak drag directions fall below the
        # solver's absolute accuracy. QR itself needs no objective rescaling.
        self.factor.value=R;self.target.value=Q.T@b
        self.problem.solve(solver=cp.MOSEK,warm_start=True,verbose=False,
            mosek_params={'MSK_IPAR_NUM_THREADS':1,'MSK_DPAR_OPTIMIZER_MAX_TIME':self.budget,
                          'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-10,
                          'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-10,
                          'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-10})
        stats=self.problem.solver_stats
        seconds=None if stats is None else stats.solve_time
        if (self.problem.status!=cp.OPTIMAL or self.theta.value is None
                or seconds is None or not np.isfinite(seconds) or seconds>self.budget):
            raise RuntimeError('Rejected failed/late MOSEK candidate: '+str(self.problem.status))
        theta=np.asarray(self.theta.value).copy()
        M=np.asarray(MarineVehicleEstimator.M_from_uv(self,cp.Constant(theta)).value)
        if (not np.isfinite(theta).all() or np.max(np.abs(M-M.T))>1e-6
                or np.linalg.eigvalsh((M+M.T)/2).min()<self.floor-1e-6
                or theta[15:].max()>1e-6 or theta[10]<self.bounds[0]-1e-6
                or theta[10]>self.bounds[1]+1e-6):
            raise RuntimeError('Rejected infeasible MOSEK candidate')
        return ConvexVehicleFit(theta,np.sqrt(np.mean((A@theta-b).reshape(-1,6)**2,axis=0)),
            float(seconds),perf_counter()-started,int(np.linalg.matrix_rank(A@self.basis)))

    def fit_robust(self, design, wrench, *, max_iterations=16, relative_tolerance=1e-4):
        """Convex Huber fit via bounded IRLS, all subproblems solved by MOSEK.

        Per-axis residual scales are estimated ONCE from the initial fit using
        MAD, then frozen. These are robust loss scales, not sensor covariance or
        posterior uncertainty. No truth, prior, or model-dependent EKF is used.
        Raises on failed convergence or total wall budget; no partial update.
        """
        if not isinstance(max_iterations,int) or max_iterations<2 or not 0<relative_tolerance<1:
            raise ValueError('Invalid robust iteration configuration')
        started=perf_counter()
        A=np.asarray(design,float);b=np.asarray(wrench,float)
        result=self.fit(A,b)
        solver_seconds=result.solver_seconds
        residual=(A@result.parameters-b).reshape(-1,6)
        scale=np.median(np.abs(residual-np.median(residual,axis=0)),axis=0)/.6744897501960817
        # Numerical zero guard only, not assumed measurement noise.
        scale=np.maximum(scale,1e-10*np.maximum(np.max(np.abs(b.reshape(-1,6)),axis=0),1))
        threshold=np.tile(1.5*scale,len(b)//6)

        def objective(parameters):
            absolute=np.abs(A@parameters-b)
            return float(np.sum(np.where(absolute<=threshold,.5*absolute**2,
                                         threshold*(absolute-.5*threshold))))

        previous=objective(result.parameters)
        for iteration in range(1,max_iterations+1):
            if perf_counter()-started>self.budget:
                raise RuntimeError('Robust fit wall budget exceeded; no candidate returned')
            residual=A@result.parameters-b
            weights=np.sqrt(np.minimum(1.,threshold/np.maximum(np.abs(residual),np.finfo(float).tiny)))
            candidate=self.fit(A*weights[:,None],b*weights)
            solver_seconds+=candidate.solver_seconds
            current=objective(candidate.parameters)
            change=abs(current-previous)/max(previous,np.finfo(float).tiny)
            increase=(current-previous)/max(previous,np.finfo(float).tiny)
            result=candidate
            if perf_counter()-started>self.budget:
                raise RuntimeError('Robust fit wall budget exceeded; no candidate returned')
            if iteration>=2 and change<relative_tolerance and increase<=relative_tolerance:
                return replace(result,residual_rms=np.sqrt(np.mean((A@result.parameters-b).reshape(-1,6)**2,axis=0)),
                    solver_seconds=solver_seconds,wall_seconds=perf_counter()-started,
                    equality_reduced_rank=int(np.linalg.matrix_rank(A@self.basis)),
                    robust_iterations=iteration,robust_relative_objective_change=change)
            previous=current
        raise RuntimeError('Robust fit did not converge; no candidate returned')
