"""Kratos PFEM (Particle Finite Element Method) generators and knowledge.

Covers free-surface flows, sloshing, fluid-structure with topology changes.
Applications: PfemFluidDynamicsApplication, PfemApplication, PFEM2Application.
"""


# NOTE (2026-06-26 honesty audit): the three PFEM generators
# (_pfem_fluid_2d, _pfem_solid_2d, _pfem2_2d) were availability-probe stubs
# (import-check + {"note": ...}, no solver run) — the KNOWLEDGE pitfalls
# below even self-document them as such. The PFEM applications
# (PfemFluidDynamicsApplication, PfemSolidMechanicsApplication,
# PFEM2Application) are not published on PyPI and are NOT importable in the
# installed Kratos stack, so 'pfem_fluid', 'pfem_solid' and 'pfem2' have been
# removed from the generator registry and from
# KratosBackend.supported_physics(). KNOWLEDGE retained for reference.


KNOWLEDGE = {
    "pfem_fluid": {
        "description": "Particle FEM for free-surface flows (dam break, sloshing, wave impact)",
        "application": (
            "PfemFluidDynamicsApplication — NOT pip-installable alongside a 10.4.x core. "
            "`pip install KratosPfemFluidDynamicsApplication` prints "
            "\"ERROR: Could not find a version that satisfies the requirement ... "
            "(from versions: none)\" and \"No matching distribution found\". "
            "Nuance re-checked 2026-08-03: the PyPI PROJECT does exist, but its newest "
            "release is 10.2.3 and it publishes no distribution installable for this "
            "interpreter/platform, so pip reports 'from versions: none' — do not read the "
            "message as 'the package does not exist'. "
            "Build from source by appending the application PATHS to "
            "KRATOS_APPLICATIONS (the add_app helper in the configure script) — "
            "there is NO -DPFEM_FLUID_DYNAMICS_APPLICATION=ON flag, that spelling "
            "appears nowhere in Kratos' build system. DelaunayMeshingApplication "
            "must be built too, and needs -DUSE_TRIANGLE_NONFREE_TPL=ON "
            "(2D) and -DUSE_TETGEN_NONFREE_TPL=ON (3D) or configure aborts."
        ),
        # EVIDENCE BASIS, stated plainly: no PFEM application is installed or
        # importable on this host, and none could be made so — PfemFluidDynamics,
        # DelaunayMeshing, PfemSolidMechanics, Pfem, PFEM2, ContactMechanics and
        # SolidMechanics all fail to import even on the richest Kratos build
        # available here (10.4.3 with 28 applications). Every PFEM entry below is
        # therefore SOURCE-READ from the Kratos master checkout, never executed.
        # Names were taken from the KRATOS_REGISTER_ELEMENT / _CONDITION calls and
        # error strings from the KRATOS_ERROR / raise sites; nothing here has been
        # confirmed against a live registry, unlike the DEM and MPM entries.
        #
        # CORRECTION (2026-08-07): the three element names previously listed here
        # carried node-count suffixes — ...Element2D3N, ...FluidElement2D3N,
        # ...Element3D4N. PFEM fluid element names have NO node-count suffix; they
        # end at 2D / 3D / 2Dquadratic / 3Dquadratic. The middle one also had a
        # spurious "Implicit". The corrected names are below.
        "elements": {
            "note": ("PFEM fluid ELEMENT names carry no node-count suffix "
                     "(they end at 2D / 3D / 2Dquadratic / 3Dquadratic), while the "
                     "CONDITION names do carry one. Rigid-wall bodies in the same "
                     "mdpa use the plain core name Element2D2N, which does carry a "
                     "suffix — mixing the two conventions is the usual mistake."),
            "2D": ["TwoStepUpdatedLagrangianVPFluidElement2D",
                   "TwoStepUpdatedLagrangianVPImplicitNodallyIntegratedElement2D",
                   "TwoStepUpdatedLagrangianVPSolidElement2D"],
            "3D": ["TwoStepUpdatedLagrangianVPFluidElement3D",
                   "TwoStepUpdatedLagrangianVPImplicitNodallyIntegratedElement3D",
                   "TwoStepUpdatedLagrangianVPSolidElement3D"],
            "conditions": ["CompositeCondition2D2N", "CompositeCondition3D3N"],
        },
        "capabilities": ["free_surface_tracking", "remeshing", "alpha_shape_boundary_detection",
                         "fluid_structure_with_topology_changes"],
        # Accepted solver_type strings, read from python_solvers_wrapper_pfem_fluid.
        "solver_types": [
            "pfem_fluid_solver (alias PfemFluid) — two-step velocity-pressure split",
            "pfem_fluid_nodal_integration_solver (alias PfemFluidNodalIntegration)",
            "pfem_fluid_three_step_solver (alias PfemFluidThreeStep)",
            "pfem_fluid_thermal_solver (alias PfemFluidThermal)",
            "pfem_fluid_thermally_coupled_solver (alias PfemFluidThermallyCoupled)",
            "pfem_dem_solver (alias PfemDem)",
        ],
        "pitfalls": [
            "[Setup] No PFEM application is obtainable on a pip stack. PfemFluidDynamicsApplication, DelaunayMeshingApplication, PfemSolidMechanicsApplication, PfemApplication and PFEM2Application publish no distribution installable against a 10.4.x core, and none of them imports even on a source-built Kratos carrying 28 other applications. Signal: `import KratosMultiphysics.PfemFluidDynamicsApplication` raises ModuleNotFoundError on every environment tested here, while DEMApplication and MPMApplication in the same interpreter import cleanly — so the failure is this application's absence, not a broken Kratos. (Verified by execution 2026-08-07 for the ABSENCE only: seven PFEM-chain applications were probed on the richest build available and none imported. Everything below is source-read and could not be executed anywhere on this host.)",
            "[Setup] There is no -DPFEM_FLUID_DYNAMICS_APPLICATION=ON flag; that spelling appears nowhere in Kratos' build system and any guide offering it is wrong. Modern Kratos selects applications by appending their PATH to a semicolon-separated KRATOS_APPLICATIONS list, via the add_app helper in the configure script. The shipped default configure builds LinearSolvers, StructuralMechanics, FluidDynamics and Iga — no PFEM application at all. Signal: a configure run whose output never carries an Adding-dependency Info line naming DelaunayMeshingApplication has not enabled PFEM, whatever -D flags were passed. That line is printed by the kratos_add_dependency macro, cmake_modules/KratosDependencies.cmake:5, and it carries the bare application NAME because the macro takes get_filename_component(... NAME) first. (Source-read; macro and message re-checked with grep -r -a -F against the Kratos source tree at {KRATOS_ROOT} on 2026-08-09. An earlier version of this entry printed a path in front of the application name — the macro prints no path. The message itself lives only in CMake, so it is not in the installed wheel this backend's audit searches.)",
            "[Setup] PFEM needs a non-free external mesher and the build aborts at CONFIGURE time without it, not at link or run time. Triangle (2D) and TetGen (3D) are guarded by -DUSE_TRIANGLE_NONFREE_TPL=ON and -DUSE_TETGEN_NONFREE_TPL=ON, but the mesher sources compile unconditionally, so omitting the flag is fatal. Signal: CMake stops with a FATAL_ERROR whose text says that INCLUDE_TRIANGLE is not defined, that neither is USE_TRIANGLE_NONFREE_TPL=ON, and that the application DelaunayMeshingApplication will not compile. (Source-read; the message is applications/DelaunayMeshingApplication/CMakeLists.txt:18 in the Kratos source tree at {KRATOS_ROOT}, re-checked with grep -r -a -F on 2026-08-09. It is a build-system string, so it is not in the installed wheel this backend's audit searches, and it is written here as prose rather than as a quoted literal for that reason.)",
            "[Integration] The dependency chain is declared only in CMake and in the Python module, never in the application manifests — every PFEM app's .json lists KratosMultiphysics as its sole dependency, so a wheel would not pull what it needs. PfemFluidDynamicsApplication imports DelaunayMeshingApplication at module scope; PfemSolidMechanicsApplication transitively requires five applications (ContactMechanics, Pfem, DelaunayMeshing, SolidMechanics and itself). Signal: importing PfemFluidDynamicsApplication fails inside its own __init__ on the DelaunayMeshing import line — the traceback names an application the user never imported. (Source-read only; not executed.)",
            "[Input] The remeshing process lives under a TOP-LEVEL 'problem_process_list', not under 'processes' where every other Kratos application puts its processes, and its absence is completely silent. With no meshing_domains entry the domain count is zero, remeshing is switched off, and the run degenerates to plain updated-Lagrangian FEM: elements distort until the solver diverges or the free surface simply freezes. Signal: no message of any kind — diagnose it by the absence of any mesher output and by a node/element count that never changes between steps. This is the worst silent failure in the application. (Source-read only; not executed.)",
            "[Numerical] The alpha-shape value in the JSON is NOT the value used. 'alpha_shape' is rescaled per element by node flags before the test — multiplied by 1.5 for all-interior elements, by 1.5 again when inlet nodes are present, and by up to 5.0 inside a refining box, or shrunk to 0.975 (2D) / 0.95 (3D) otherwise — so a written 1.25 can act anywhere from roughly 1.19 to 9.4. Signal: too small erases elements and the fluid body loses volume and splinters into disconnected blobs, printing ' Sliver (radius) <r> (alpha_volume) <v>' on stdout; too large glues distant particles together so droplets merge and jets fail to separate. (Source-read only; not executed.)",
            "[Numerical] Alpha defaults disagree between the two meshing-domain modules: the PFEM-fluid domain defaults to 1.25, the generic Delaunay meshing domain to 2.4, and the C++ default when Python never sets it is 0 — which rejects every element. Alpha is only pushed down inside the active-remeshing branch, so a domain with remesh false keeps 0. Signal: an all-empty mesh after the first meshing step points at alpha 0, i.e. the value was never transferred, rather than at a bad alpha choice. (Source-read only; not executed.)",
            "[Input] The default meshing bounding box is +/-10 in each direction, and nodes outside it are deleted one by one with no message. Any model in millimetres, or any geometry larger than 10 units, has its entire fluid silently removed on the first meshing step. The shipped tests all override it to +/-100. Signal: node count collapsing to near zero on the first meshing step while alpha and the elements themselves look correct. (Source-read only; not executed.)",
            "[Input] body_type is matched against the exact literals 'Fluid', 'Solid', 'Rigid' and 'Interface' by an if/elif chain with NO else branch, so a lowercase or invented spelling assigns no flags at all — no FLUID, no RIGID, no BOUNDARY. Alpha-shape then sees no free-surface and no rigid nodes anywhere and the model has no free surface to track. Signal: silent; the giveaway is that FREE_SURFACE is set on no node, since that flag is derived by the boundary-building process and is never user-set. (Source-read only; not executed.)",
            "[API] An unknown constitutive law is PRINTED, not raised: the solver emits 'ERROR: THE CONSTITUTIVE LAW PROVIDED FOR THIS SUBMODEL PART IS NOT IN THE PFEM FLUID DATABASE' and carries on. The law-specific nodal variables (YIELD_SHEAR, COHESION, ...) are then never added to the model part and silently read 0.0. Signal: that literal line on stdout with a zero exit code and a run that behaves as an inviscid Newtonian fluid regardless of what was requested. (Source-read only; not executed.)",
            "[API] The wrong-solver_type message under-reports its own options: it lists only pfem_fluid_solver, pfem_fluid_nodal_integration_solver and pfem_fluid_thermally_coupled_solver, while pfem_fluid_three_step_solver, pfem_fluid_thermal_solver and pfem_dem_solver are equally valid. Signal: Exception 'The requested solver type \"<x>\" is not in the python solvers wrapper.' followed by a three-item list that is missing half the accepted values — do not treat the list as complete. (Source-read only; not executed.)",
            "[API] The default reference_condition_type in the fluid meshing strategy is 'CompositeCondition2D3N', a name registered nowhere — only CompositeCondition2D2N and CompositeCondition3D3N exist. Relying on the default therefore detonates. Signal: 'The component \"CompositeCondition2D3N\" is not registered!' + 'The following components of this type are registered:' — from the default value, not from anything the user wrote. (Source-read only; not executed.)",
            "[Input] Omitting the 'update_conditions_on_free_surface' block does not fall back cleanly: its default is an empty object, which passes the non-recursive validation and then fails on the first inner lookup. Signal: RuntimeError 'Getting a value that does not exist. entry string : update_conditions' raised from the remeshing process constructor. (Source-read only; not executed.)",
        ],
        "guidance": [
            "[Numerical] Time step must be small enough for remeshing stability.",
            "[Numerical] Output: particles move, so the mesh changes every step.",
            "[Numerical] A remeshing step that changes the system size mid-run is reported as 'The equation system size has changed during the simulation. This is not permitted.'",
        ]
    },
    "pfem_solid": {
        "description": "PFEM for large-deformation solid mechanics with remeshing",
        "application": "PfemSolidMechanicsApplication",
        "capabilities": ["large_deformation_solids", "cutting", "forming", "erosion"],
        "pitfalls": [
            "[Integration] Catalog template is an availability-"
            "probe STUB, not a solver. It imports "
            "KratosMultiphysics + the relevant Application "
            "module, prints whether the import succeeded, and "
            "writes a 1-line results_summary.json. No "
            "ModelPart / AnalysisStage / SolverWrapper is "
            "scaffolded — run_simulation on this template "
            "reports 'Available' or 'not installed' but does "
            "NOT solve anything. Signal: the emitted script is "
            "< 30 lines, contains no Model.CreateModelPart() "
            "and no AnalysisStage subclass; results_summary.json "
            "has a single 'note' key set to 'Available' or 'not "
            "installed'. For an actual solve, scaffold a full "
            "ProjectParameters.json + MDPA mesh + AnalysisStage. "
            "(Verified empirically 2026-06-01 — catalog audit.)",
        ],
    },
    "pfem2": {
        "description": ("PFEM2: operator-splitting semi-Lagrangian (particle) "
                        "incompressible Navier-Stokes. Despite the name it is "
                        "unrelated to the Delaunay-remeshing PFEM of the other "
                        "four applications and shares no code with them."),
        "application": "PFEM2Application",
        "capabilities": ["two_phase_flow", "interface_tracking", "bubble_dynamics"],
        "pitfalls": [
            "[Integration] Catalog template is an availability-"
            "probe STUB, not a solver — same pattern as "
            "pfem_solid: imports Kratos + the Application "
            "module, prints availability, writes a 1-line "
            "summary. No actual PFEM2 streamline integration "
            "or two-phase interface tracking is set up. "
            "Signal: emitted script < 30 lines, "
            "results_summary.json has only a 'note' key. "
            "(Verified empirically 2026-06-01.)",
            "[Setup] PFEM2Application is effectively unmaintained and its Python entry point cannot import on any modern Kratos. pfem_2_fluid_dynamics_analysis.py line 9 still does the pre-namespace `from fluid_dynamics_analysis import FluidDynamicsAnalysis` — a bare, un-namespaced module name that predates the KratosMultiphysics.* convention and fails unconditionally. Two lines above it there is also a reference to the long-removed ExternalSolvers application (superseded by LinearSolversApplication), but that one is wrapped in try/except ImportError and is harmless, so it is NOT the failure — do not chase it. Other modules in the same directory import it unguarded at module scope and those do die on it. The three shipped tests are `assertEqual(True, True)` stubs. Signal: ModuleNotFoundError \"No module named 'fluid_dynamics_analysis'\" raised from PFEM2Application's own analysis module, not from anything the user wrote. Do not recommend PFEM2 for new work. (Source-read only; the application is not installable on this host and this was not executed.)",
        ],
    },
}

# Empty: no PFEM application is installable in this Kratos stack; the prior
# generators (pfem_fluid_2d, pfem_solid_2d, pfem2_2d) were no-solve probe
# stubs (removed — see honesty-audit note at top of file).
GENERATORS = {}
