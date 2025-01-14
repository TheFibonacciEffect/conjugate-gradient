# Iterative Solvers

## Description

1. Conjugate Gradient (CG) [1] for the discrete Laplace operator [2]. The application of the discrete Laplace operator $`A`$ for arbitrary dimension $`d=1,2,3,4,...`$ should be implemented as a vector function, i.e. as function `laplace(v,u)` with $`v=A\, u`$. In particular, do not implement $`A`$ itself but the application on a given vector.

2. Mixed precision CG: Use *preconditioning* (chapter 12 in [1]) to
design an algorithm that performs the dominant part of operations on the GPU in single precision (`float`), but yields the result in double precision (`double`). The double precision part can be performed on the CPU. Tip: Use the complete `float`-CG as preconditioning matrix $`M^{-1}`$. Show analytically that this $`M^{-1}`$ is symmetric positive-definite, if $A$ is.

3. Use the power methode/power iteration in together with CG, to determine the smallest and the largest eigenvalue ($`\lambda_{min}`$, $`\lambda_{max}`$) (see e.g. [3])
```math
    \begin{aligned}
    & q^{(0)}  \;\mathrm{arbitrary} \\
    & \mathrm{for} \; k=1,2,...  \\
    & \quad z^{(k)}=A q^{(k-1)} \\
    & \quad q^{(k)}=z^{(k)}/||z^{(k)}|| \\
    & \quad \lambda^{(k)}={q^{(k)}}^{T} q^{(k)} \\
    & \mathrm{end}
    \end{aligned}
```
with $`|\lambda_{max} - \lambda^{(k)}|\to 0`$.

4. Determine the necessary number of CG iterations for a given precision (reduction of residue norm) and compare it to the expected dependence of the convergence on the condition number $`\kappa=\lambda_{max}/\lambda_{min}`$.

## Preconditioner

- implement CG with a Jacobi-preconditioner and the mixed precision CG, described above
- in first case: determine the improvement for convergence

## Multigrid

- implement the two grid correction scheme TG in [4] with Jacobi or Gauss-Seidel as smoother/relaxation method
- compare the convergence and time to solution of TG with CG
- use TG as preconditioner for CG

[1] Jonathan Richard Shewchuk, "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" [http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

[2] Generators for large sparse systems [http://people.physik.hu-berlin.de/~leder/cp3/laplace.pdf](http://people.physik.hu-berlin.de/~leder/cp3/laplace.pdf)

[3] Gene H. Golub,  Charles F. Van Loan "Matrix Computations", Johns Hopkins University Press, 1989

[4] Briggs, William & Henson, Van & McCormick, Steve. (2000). "A Multigrid Tutorial, 2nd Edition" [https://www.researchgate.net/publication/220690328_A_Multigrid_Tutorial_2nd_Edition](https://www.researchgate.net/publication/220690328_A_Multigrid_Tutorial_2nd_Edition)

[5] Tatebe, Osamu. (1995). "The Multigrid Preconditioned Conjugate Gradient Method" [https://www.researchgate.net/publication/2818681_The_Multigrid_Preconditioned_Conjugate_Gradient_Method](https://www.researchgate.net/publication/2818681_The_Multigrid_Preconditioned_Conjugate_Gradient_Method)
