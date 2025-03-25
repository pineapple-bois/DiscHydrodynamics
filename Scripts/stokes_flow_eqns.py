import sympy as sp


class StokesEquations:
    def __init__(self):
        # Spatial coordinates and position vector
        self.x, self.y = sp.symbols('x y')
        self.r = sp.Matrix([self.x, self.y])

        # Velocity field components as functions of x and y
        self.v_x = sp.Function('v_x')(self.r[0], self.r[1])
        self.v_y = sp.Function('v_y')(self.r[0], self.r[1])
        self.v = sp.Matrix([self.v_x, self.v_y])

        # Pressure field
        self.p_field = sp.Function('p')(self.r[0], self.r[1])

        # Director components for anisotropy (to be set along x-axis later)
        self.n_x, self.n_y = sp.symbols('n_x n_y', real=True)
        self.n = (self.n_x, self.n_y)

        # Viscosity coefficients
        self.nu1, self.nu2, self.nu3, self.nu4, self.nu5 = sp.symbols('nu1 nu2 nu3 nu4 nu5', real=True)
        self.nu_bar = sp.Symbol('nubar')

        # Compute velocity gradient: dv[j,k] = ∂v_k/∂x_j
        self.dv = sp.Matrix([
            [self.v_x.diff(self.x), self.v_y.diff(self.x)],
            [self.v_x.diff(self.y), self.v_y.diff(self.y)]
        ])

        # External force components
        self.f_x = sp.Function('f_x')(self.r[0], self.r[1])
        self.f_y = sp.Function('f_y')(self.r[0], self.r[1])
        self.f_ext = sp.Matrix([self.f_x, self.f_y])

        # Anisotropic friction coefficients (for a diagonal friction tensor)
        self.m_parallel, self.m_perp = sp.symbols('m_∥ m_⊥', real=True, positive=True)

        # Dimensionless parameters (to be used in non-dimensional)
        self.zeta1, self.zeta2 = sp.symbols('zeta_1 zeta_2', real=True)
        self.alpha_par_sq, self.alpha_perp_sq = sp.symbols('alpha_∥^2 alpha_⊥^2', real=True)

    def build_nu_tensor(self):
        """Builds the 4th-order viscosity tensor as a dictionary with keys (i,j,k,l)."""
        self.nu_tensor = {}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        val = 0
                        # Term 1: isotropic-like part
                        val += self.nu2 * (
                                sp.KroneckerDelta(i, k) * sp.KroneckerDelta(j, l) +
                                sp.KroneckerDelta(i, l) * sp.KroneckerDelta(j, k)
                        )
                        # Term 2:
                        val += 2 * (self.nu1 + self.nu2 - 2 * self.nu3) * self.n[i] * self.n[j] * self.n[k] * self.n[l]
                        # Term 3:
                        val += (self.nu4 - self.nu2) * (sp.KroneckerDelta(i, j) * sp.KroneckerDelta(k, l))
                        # Term 4:
                        val += (self.nu3 - self.nu2) * (
                                self.n[i] * self.n[k] * sp.KroneckerDelta(j, l) +
                                self.n[i] * self.n[l] * sp.KroneckerDelta(j, k) +
                                self.n[j] * self.n[k] * sp.KroneckerDelta(i, l) +
                                self.n[j] * self.n[l] * sp.KroneckerDelta(i, k)
                        )
                        # Term 5:
                        val += (self.nu5 - self.nu4 + self.nu2) * (
                                sp.KroneckerDelta(i, j) * self.n[k] * self.n[l] +
                                sp.KroneckerDelta(k, l) * self.n[i] * self.n[j]
                        )
                        self.nu_tensor[(i, j, k, l)] = sp.simplify(val)
        return self.nu_tensor

    def build_sigma_D(self):
        """
        Constructs the dissipative (deviatoric) stress tensor σ_D,
        substituting n_x=1 and n_y=0 (director along x-axis).
        """
        if not hasattr(self, 'nu_tensor'):
            self.build_nu_tensor()
        sigma_D = sp.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                expr = 0
                for k in range(2):
                    for l in range(2):
                        expr += -self.nu_tensor[(i, j, k, l)] * self.dv[l, k]
                sigma_D[i, j] = sp.simplify(expr.subs({self.n_x: 1, self.n_y: 0}))
        self.sigma_D = sp.simplify(sigma_D)
        return self.sigma_D

    def build_sigma_total(self):
        """Builds the total stress tensor: σ_total = p*δ_ij + σ_D."""
        if not hasattr(self, 'sigma_D'):
            self.build_sigma_D()
        sigma_total = sp.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                sigma_total[i, j] = self.p_field * sp.KroneckerDelta(i, j) + self.sigma_D[i, j]
        self.sigma_total = sigma_total
        return self.sigma_total

    def build_div_sigma(self):
        """Computes the divergence of σ_total: (∇·σ)_i = Σ_j ∂σ_total[i,j]/∂(coordinate_j)."""
        if not hasattr(self, 'sigma_total'):
            self.build_sigma_total()
        coords = (self.x, self.y)
        div_sigma = sp.Matrix([
            sum(self.sigma_total[i, j].diff(coords[j]) for j in range(2))
            for i in range(2)
        ])
        self.div_sigma = sp.simplify(div_sigma)
        return self.div_sigma

    def adjust_div_sigma(self):
        """
        Adjusts the divergence of σ_total by adding an operator based on the gradient
        of the velocity divergence. This step incorporates the term (nu3+nu5)∇(∇·v).
        """
        # Divergence of v
        div_v = sp.diff(self.v_x, self.x) + sp.diff(self.v_y, self.y)
        grad_div_v = sp.Matrix([sp.diff(div_v, self.x), sp.diff(div_v, self.y)])
        operator_to_subtract = sp.expand((self.nu3 + self.nu5) * grad_div_v)
        div_sigma_expand = sp.expand(self.div_sigma)
        self.div_sigma_final = sp.simplify(div_sigma_expand + operator_to_subtract)
        # Optionally, collect terms in second derivatives for clarity
        self.div_sigma_final = sp.Matrix([
            sp.collect(self.div_sigma_final[0], sp.diff(self.v_x, (self.x, 2))),
            sp.collect(self.div_sigma_final[1], sp.diff(self.v_y, (self.y, 2)))
        ])
        return self.div_sigma_final

    def build_friction(self):
        """Constructs the anisotropic friction force using a diagonal friction tensor."""
        M_diag = sp.Matrix([
            [self.m_parallel ** 2, 0],
            [0, self.m_perp ** 2]
        ])
        self.M_diag = M_diag
        self.friction = M_diag * self.v
        return self.friction

    def build_momentum_eq(self):
        """
        Constructs the momentum balance equation:
          ∇_j σ_total[i,j] + M_ij v_j = f_ext[i].
        """
        if not hasattr(self, 'div_sigma'):
            self.build_div_sigma()
        self.adjust_div_sigma()
        if not hasattr(self, 'friction'):
            self.build_friction()
        self.momentum_eq = sp.Eq(self.div_sigma_final + self.friction, self.f_ext)
        return self.momentum_eq

    def nondimensionalize(self):
        """
        Applies nondimensional substitutions to the momentum equation.
        Substitutions:
          zeta_1 = (2*nu1 + nu2 - nu3 - nu4 + nu5) / nu3
          zeta_2 = (nu2 - nu3 + nu4 - nu5) / nu3
          α_∥² = m_parallel²/nu3,  α_⊥² = m_perp²/nu3.
        """
        if not hasattr(self, 'momentum_eq'):
            self.build_momentum_eq()

        subs_dimensionless = {
            (2 * self.nu1 / self.nu3 + self.nu2 / self.nu3 - 1 - self.nu4 / self.nu3 + self.nu5 / self.nu3): self.zeta1,
            (self.nu2 / self.nu3 - 1 + self.nu4 / self.nu3 - self.nu5 / self.nu3): self.zeta2,
            (self.m_parallel ** 2 / self.nu3): self.alpha_par_sq,
            (self.m_perp ** 2 / self.nu3): self.alpha_perp_sq
        }

        # Expand and isolate x eqn components:
        expr_x = sp.expand(self.momentum_eq.lhs[0])
        modified_expr_x = 0
        for term in sp.Add.make_args(expr_x):
            if term.has(self.m_parallel ** 2) or term.has(self.v_x.diff(self.x, 2)):
                modified_expr_x += term / self.nu3
            else:
                modified_expr_x += term
        modified_expr_x_collect = sp.collect(modified_expr_x,
                                             sp.diff(self.v_x, (self.x, 2))
                                             )

        # Expand and isolate y eqn components:
        expr_y = sp.expand(self.momentum_eq.lhs[1])
        modified_expr_y = 0
        for term in sp.Add.make_args(expr_y):
            if term.has(self.m_perp ** 2) or term.has(self.v_y.diff(self.y, 2)):
                modified_expr_y += term / self.nu3
            else:
                modified_expr_y += term
        modified_expr_y_collect = sp.collect(modified_expr_y,
                                             sp.diff(self.v_y, (self.y, 2))
                                             )

        # Rebuild interim equations
        self.momentum_eq_dimless = sp.Eq(
            sp.Matrix([modified_expr_x_collect, modified_expr_y_collect]),
            self.f_ext
        )
        # Perform substitutions
        self.momentum_eq_dimless = self.momentum_eq_dimless.subs(subs_dimensionless)

        # Expand and isolate row 0 (the x‐component)
        expr_x = sp.expand(self.momentum_eq_dimless.lhs[0])
        modified_expr_x = 0
        for term in sp.Add.make_args(expr_x):
            if term.has(self.alpha_par_sq) or term.has(self.zeta1):
                # Divide these particular terms by nu3
                modified_expr_x += self.nu3 * term
            else:
                # Keep the rest unchanged
                modified_expr_x += term

        modified_expr_x_collect = sp.collect(modified_expr_x, self.nu3)

        # Expand and isolate row 0 (the x‐component)
        expr_y = sp.expand(self.momentum_eq_dimless.lhs[1])
        modified_expr_y = 0
        for term in sp.Add.make_args(expr_y):
            if term.has(self.alpha_perp_sq) or term.has(self.zeta2):
                # Divide these particular terms by nu3
                modified_expr_y += self.nu3 * term
            else:
                # Keep the rest unchanged
                modified_expr_y += term

        modified_expr_y_collect = sp.collect(modified_expr_y, self.nu3)

        built_eq_lhs = sp.Matrix([
            modified_expr_x_collect,
            modified_expr_y_collect
        ])
        self.built_eq_final = sp.Eq(built_eq_lhs, self.f_ext)

        return self.built_eq_final

    def reverse_substitutions(self):
        """
        Reverts the nondimensional substitutions, switching back to the original viscosity coefficients.
        """
        if not hasattr(self, 'momentum_eq_dimless'):
            self.nondimensionalize()
        subs_original = {
            self.zeta1: (2 * self.nu1 + self.nu2 - self.nu3 - self.nu4 + self.nu5) / self.nu3,
            self.zeta2: (self.nu2 - self.nu3 + self.nu4 - self.nu5) / self.nu3,
            self.alpha_par_sq: self.m_parallel ** 2 / self.nu3,
            self.alpha_perp_sq: self.m_perp ** 2 / self.nu3
        }
        self.momentum_eq_original = sp.simplify(self.momentum_eq_dimless.subs(subs_original))
        return self.momentum_eq_original

    def isotropic_case(self):
        """
        Returns the Brinkman equation (isotropic limit) by substituting
          nu1 = nu2 = nu3 = nu4 = nu and nu5 = 0.
        """
        if not hasattr(self, 'momentum_eq'):
            self.build_momentum_eq()
        nu = sp.Symbol('nu')
        subs_brinkman = {
            self.nu1: nu,
            self.nu2: nu,
            self.nu3: nu,
            self.nu4: nu,
            self.nu5: 0
        }
        self.momentum_eq_brinkman = sp.simplify(self.momentum_eq.subs(subs_brinkman))
        return self.momentum_eq_brinkman

    def get_xy_equations(self, nondimensional=False):
        """
        Returns the momentum balance equations for the x- and y-components separately.
        If `nondimensional=True`, the nondimensional form is returned.
        """
        if nondimensional:
            if not hasattr(self, 'momentum_eq_dimless'):
                self.nondimensionalize()
            eq = self.momentum_eq_dimless
        else:
            if not hasattr(self, 'momentum_eq'):
                self.build_momentum_eq()
            eq = self.momentum_eq
        x_eq = sp.Eq(eq.lhs[0], self.f_ext[0])
        y_eq = sp.Eq(eq.lhs[1], self.f_ext[1])
        return x_eq, y_eq
