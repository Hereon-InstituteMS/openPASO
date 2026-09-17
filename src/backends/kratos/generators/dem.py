"""Kratos DEM (Discrete Element Method) generators and knowledge.

Uses the REAL Kratos DEMApplication — NOT standalone numpy/scipy.
The generator is a FILE WRITER: it emits the four files DEMApplication reads
(<problem_name>DEM.mdpa, <problem_name>DEM_FEM_boundary.mdpa, MaterialsDEM.json,
ProjectParametersDEM.json) plus input.py, the entry point that calls
DEMAnalysisStage. Both halves are executed by
scripts/audit_two_stage_templates.py; the generator exiting 0 is NOT evidence
that the deck it wrote runs, and for a long time it did not.
"""


def _dem_2d_kratos(params: dict) -> str:
    n_particles = int(params.get("n_particles", 64))
    radius = float(params.get("radius", 0.01))
    density = float(params.get("density", 2650.0))
    young_modulus = float(params.get("E", 1.0e7))
    poisson = float(params.get("nu", 0.25))
    restitution = float(params.get("restitution", 0.5))
    friction = float(params.get("friction", 0.4))
    gravity_y = float(params.get("gravity", -9.81))
    domain_x = float(params.get("domain_x", 0.30))
    domain_y = float(params.get("domain_y", 0.40))
    drop_height = float(params.get("drop_height", 0.05))
    T_end = float(params.get("T_end", 0.25))
    dt = params.get("dt", None)
    dt_safety = float(params.get("dt_safety_factor", 0.02))
    problem_name = str(params.get("problem_name", "granular"))
    # Clamp: OMP_NUM_THREADS=0 is not "use the default", it is undefined
    # behaviour — some runtimes abort, others silently serialise. A user
    # passing omp_threads=0 meaning "let OpenMP decide" would get a deck that
    # fails for a reason unrelated to anything they were trying to simulate,
    # and the DEM run reports particle loss silently, so a crash here is the
    # kinder outcome of the two. Flagged in review of PR #53.
    threads = max(1, int(params.get("omp_threads", 2)))

    header = f'''\
"""Kratos DEM — 2D granular settling — writes a complete DEMApplication deck.

This script is a FILE WRITER. It emits

    {problem_name}DEM.mdpa               spheres (CylinderParticle2D)
    {problem_name}DEM_FEM_boundary.mdpa  rigid walls (RigidFace3D3N)
    MaterialsDEM.json                    materials + material_relations + table
    ProjectParametersDEM.json            FLAT DEM schema
    input.py                             the entry point that calls Kratos

and then exits. Run `python input.py` to perform the analysis.

Schema notes (each verified by execution against DEMApplication 10.4.3):
  * ProjectParametersDEM.json is FLAT. There is no "problem_data" block and no
    "solver_settings.solver_type"/"time_stepping". DEM reads
    solver_settings.strategy, a python module name inside DEMApplication.
  * The mdpa filenames are problem_name + a FIXED tag ("DEM",
    "DEM_FEM_boundary", ...) with no separator. A missing file is not an
    error -- the run completes on an empty model -- so the names must be right.
  * MaterialsDEM.json needs materials / material_relations /
    material_assignation_table. Friction is STATIC_FRICTION + DYNAMIC_FRICTION
    per contact PAIR in material_relations; PARTICLE_FRICTION does not exist.
  * MaxTimeStep is the only key that sets dt and it is used verbatim. There is
    no stability check, so dt is sized here from the Rayleigh estimate.
"""
import json
import math

# ---- problem definition -------------------------------------------------
problem_name = "{problem_name}"
n_particles  = {n_particles}
radius       = {radius}
density      = {density}          # kg/m^3  -> PARTICLE_DENSITY
young_modulus = {young_modulus}   # Pa      -> YOUNG_MODULUS
poisson      = {poisson}
restitution  = {restitution}      # -> COEFFICIENT_OF_RESTITUTION (contact pair)
friction     = {friction}         # -> STATIC_FRICTION / DYNAMIC_FRICTION (pair)
gravity_y    = {gravity_y}
domain_x     = {domain_x}
domain_y     = {domain_y}
drop_height  = {drop_height}
T_end        = {T_end}
dt_override  = {dt!r}
dt_safety    = {dt_safety}
omp_threads  = {threads}
'''

    body = r'''
# ---- time step ----------------------------------------------------------
# Kratos DEM never checks stability: MaxTimeStep is used verbatim, and a step
# above the limit blows the contacts up silently (particles leave the bounding
# box and are deleted without a message). Size it from the Rayleigh surface-wave
# criterion, which is the standard DEM estimate:
#
#     dt_R = pi * R * sqrt(rho / G) / (0.1631 * nu + 0.8766),   G = E / (2(1+nu))
#
# and take a small fraction of it. Measured for the default parameters below:
# dt_R = 8.8e-4 s, while the deck is already unstable at 1e-4 (12 of 16
# particles lost) and stable at 5e-5 -- i.e. the true limit is near 0.07*dt_R,
# so the Rayleigh value is optimistic by more than an order of magnitude for the
# viscous-Hertz law. dt_safety = 0.02 sits a factor ~3 below the measured limit.
shear_modulus = young_modulus / (2.0 * (1.0 + poisson))
dt_rayleigh = (math.pi * radius * math.sqrt(density / shear_modulus)
               / (0.1631 * poisson + 0.8766))
dt = float(dt_override) if dt_override is not None else dt_safety * dt_rayleigh
n_steps = max(1, int(round(T_end / dt)))

# ---- particle layout: a block dropped onto the floor --------------------
spacing = 2.2 * radius
cols = max(1, int(math.floor((domain_x - 2.0 * spacing) / spacing)))
cols = min(cols, max(1, int(math.ceil(math.sqrt(n_particles)))))
rows = int(math.ceil(n_particles / cols))
block_w = (cols - 1) * spacing
x0 = 0.5 * (domain_x - block_w)

positions = []
for i in range(n_particles):
    r, c = divmod(i, cols)
    positions.append((x0 + c * spacing,
                      drop_height + radius + r * spacing,
                      0.0))

top_y = max(p[1] for p in positions)
if top_y + radius >= domain_y:
    raise SystemExit(
        "particle block (top y=%.4f) does not fit under domain_y=%.4f; "
        "reduce n_particles/radius or raise domain_y" % (top_y + radius, domain_y))

# ---- spheres mdpa: <problem_name>DEM.mdpa -------------------------------
# The element name is CylinderParticle2D: Kratos DEM is not 3D-only, and a
# genuine 2D disc is the right element for a planar problem. There is no
# SphericParticle2D. RADIUS is a CORE nodal variable and is the one mandatory
# per-node input; a particle whose RADIUS was never written keeps a silent 0.
lines = ["Begin ModelPartData", "End ModelPartData", "",
         "Begin Properties 1", "End Properties", "", "Begin Nodes"]
for i, (x, y, z) in enumerate(positions, 1):
    lines.append("  %d %.10f %.10f %.10f" % (i, x, y, z))
lines += ["End Nodes", "", "Begin Elements CylinderParticle2D"]
for i in range(1, n_particles + 1):
    lines.append("  %d 1 %d" % (i, i))
lines += ["End Elements", "", "Begin NodalData RADIUS"]
for i in range(1, n_particles + 1):
    lines.append("  %d 0 %.10f" % (i, radius))
lines += ["End NodalData", "", "Begin SubModelPart DEMParts_Body",
          "  Begin SubModelPartNodes"]
lines += ["    %d" % i for i in range(1, n_particles + 1)]
lines += ["  End SubModelPartNodes", "  Begin SubModelPartElements"]
lines += ["    %d" % i for i in range(1, n_particles + 1)]
lines += ["  End SubModelPartElements", "End SubModelPart", ""]
with open(problem_name + "DEM.mdpa", "w") as f:
    f.write("\n".join(lines))

# ---- walls mdpa: <problem_name>DEM_FEM_boundary.mdpa --------------------
# Walls are CONDITIONS, and only the 3D face spellings are registered
# (RigidFace2D2N does not exist), so a 2D wall is a 3D face given a thickness
# in z. Each wall is a quad split into two RigidFace3D3N triangles.
zlo, zhi = -5.0 * radius, 5.0 * radius
walls = {
    "floor": [(0.0, 0.0, zlo), (domain_x, 0.0, zlo),
              (domain_x, 0.0, zhi), (0.0, 0.0, zhi)],
    "left":  [(0.0, 0.0, zlo), (0.0, domain_y, zlo),
              (0.0, domain_y, zhi), (0.0, 0.0, zhi)],
    "right": [(domain_x, 0.0, zlo), (domain_x, domain_y, zlo),
              (domain_x, domain_y, zhi), (domain_x, 0.0, zhi)],
}
wlines = ["Begin ModelPartData", "End ModelPartData", "",
          "Begin Properties 2", "End Properties", "", "Begin Nodes"]
node_id, cond_id = 0, 0
wall_nodes, wall_conds = {}, {}
node_lines, cond_lines = [], []
for name, quad in walls.items():
    ids = []
    for (x, y, z) in quad:
        node_id += 1
        ids.append(node_id)
        node_lines.append("  %d %.10f %.10f %.10f" % (node_id, x, y, z))
    wall_nodes[name] = ids
    cids = []
    for tri in ((0, 1, 2), (0, 2, 3)):
        cond_id += 1
        cids.append(cond_id)
        cond_lines.append("  %d 2 %d %d %d"
                          % (cond_id, ids[tri[0]], ids[tri[1]], ids[tri[2]]))
    wall_conds[name] = cids
wlines += node_lines + ["End Nodes", "", "Begin Conditions RigidFace3D3N"]
wlines += cond_lines + ["End Conditions", ""]
for name in walls:
    wlines += ["Begin SubModelPart DEM-FEM-Wall_%s" % name,
               "  Begin SubModelPartData", "  End SubModelPartData",
               "  Begin SubModelPartNodes"]
    wlines += ["    %d" % i for i in wall_nodes[name]]
    wlines += ["  End SubModelPartNodes", "  Begin SubModelPartConditions"]
    wlines += ["    %d" % i for i in wall_conds[name]]
    wlines += ["  End SubModelPartConditions", "End SubModelPart", ""]
with open(problem_name + "DEM_FEM_boundary.mdpa", "w") as f:
    f.write("\n".join(wlines))

# ---- MaterialsDEM.json --------------------------------------------------
# Three mandatory top-level keys. Per-particle data goes in "materials";
# everything about a CONTACT (restitution, friction, the constitutive law name)
# goes in "material_relations", keyed by the pair of material ids. Every
# sub model part that carries entities needs a row in the assignation table --
# omitting the wall row fails deep inside the solver with a PROPERTIES_ID error.
contact_pair_variables = {
    "COEFFICIENT_OF_RESTITUTION": restitution,
    "STATIC_FRICTION": friction,
    "DYNAMIC_FRICTION": friction,
    "FRICTION_DECAY": 500.0,
    "ROLLING_FRICTION": 0.01,
    "ROLLING_FRICTION_WITH_WALLS": 0.01,
    "DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME": "DEM_D_Hertz_viscous_Coulomb2D",
}
materials = {
    "materials": [
        {"material_name": "granular", "material_id": 1,
         "Variables": {"PARTICLE_DENSITY": density,
                       "YOUNG_MODULUS": young_modulus,
                       "POISSON_RATIO": poisson}},
        {"material_name": "wall", "material_id": 2,
         "Variables": {"PARTICLE_DENSITY": 7850.0,
                       "YOUNG_MODULUS": 100.0 * young_modulus,
                       "POISSON_RATIO": 0.30}},
    ],
    "material_relations": [
        {"material_names_list": ["granular", "granular"],
         "material_ids_list": [1, 1],
         "Variables": dict(contact_pair_variables)},
        {"material_names_list": ["granular", "wall"],
         "material_ids_list": [1, 2],
         "Variables": dict(contact_pair_variables)},
    ],
    "material_assignation_table": [
        ["SpheresPart.DEMParts_Body", "granular"],
    ] + [["RigidFacePart.DEM-FEM-Wall_%s" % n, "wall"] for n in walls],
}
with open("MaterialsDEM.json", "w") as f:
    json.dump(materials, f, indent=2)

# ---- ProjectParametersDEM.json (FLAT) -----------------------------------
# Unknown top-level keys and unknown keys inside solver_settings are rejected by
# ValidateAndAssignDefaults, so this list is exactly the DEM vocabulary.
# FinalTime -- not problem_data.end_time -- decides when the run stops.
project_parameters = {
    "problem_name": problem_name,
    "Dimension": 2,
    "FinalTime": T_end,
    "MaxTimeStep": dt,
    "OutputTimeStep": max(dt, T_end / 20.0),
    "GravityX": 0.0,
    "GravityY": gravity_y,
    "GravityZ": 0.0,
    "ElementType": "CylinderPartDEMElement2D",
    "TranslationalIntegrationScheme": "Symplectic_Euler",
    "RotationalIntegrationScheme": "Direct_Integration",
    "RotationOption": True,
    "RollingFrictionOption": False,
    "dem_inlet_option": False,
    "BoundingBoxOption": True,
    "AutomaticBoundingBoxOption": False,
    "BoundingBoxEnlargementFactor": 1.0,
    "BoundingBoxMinX": -domain_x,
    "BoundingBoxMinY": -domain_y,
    "BoundingBoxMinZ": -1.0,
    "BoundingBoxMaxX": 2.0 * domain_x,
    "BoundingBoxMaxY": 2.0 * domain_y,
    "BoundingBoxMaxZ": 1.0,
    "NeighbourSearchFrequency": 10,
    "SearchTolerance": 0.5 * radius,
    # Output goes through DEM's own GiD writer, into <problem_name>_Post_Files.
    # The two alternatives do not work on a stock build:
    #   post_vtk_option    -> DEMApplication.dem_vtk_output imports pyevtk, which
    #                         Kratos does not depend on and does not install.
    #   the core vtk_output_process, attached to SpheresPart -> its Check() runs
    #                         before DEM populates that part's ProcessInfo, so
    #                         output_control_type "step" raises 'STEP not found
    #                         in process info of SpheresPart' and "time" raises
    #                         the same for TIME.
    "do_print_results_option": True,
    "post_gid_option": True,
    "post_vtk_option": False,
    "PostVelocity": True,
    "PostRadius": True,
    "PostTotalForces": True,
    "solver_settings": {
        "strategy": "sphere_strategy",
        "material_import_settings": {"materials_filename": "MaterialsDEM.json"},
    },
}
with open("ProjectParametersDEM.json", "w") as f:
    json.dump(project_parameters, f, indent=2)

# ---- input.py: the entry point ------------------------------------------
INPUT_PY = """# Kratos DEM entry point. Reads ProjectParametersDEM.json from this directory.
#
# Cap OpenMP threads BEFORE importing Kratos: a few-hundred-particle DEM step is
# dominated by parallel-region spin overhead, so more threads is slower, not
# faster. Measured on this deck, 2000 steps: 98 s at the machine default (32
# threads) against 1.5 s at 2, with identical results to the last digit.
#
# Treat the RATIO as indicative, not as a specification. It was taken on a
# shared 32-core box carrying a load average around 43, where oversubscribed
# spin-wait is at its worst; on an idle machine the gap is smaller. What does
# not depend on the load is the direction and the reason -- the parallel region
# costs more to enter than the work inside it is worth at this problem size --
# and that is independently recorded elsewhere in this catalog from a separate
# 2026-06-12 measurement on a different case.
#
# setdefault, so an explicit OMP_NUM_THREADS in the environment still wins.
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "__THREADS__")

import KratosMultiphysics
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage


class GranularSettling(DEMAnalysisStage):
    # DEMAnalysisStage.Finalize() deletes its model parts, and holding a python
    # reference to one across that call segfaults. Read what is needed here,
    # before delegating.
    def Finalize(self):
        spheres = self.spheres_model_part
        walls = self.rigid_face_model_part
        ys = [n.Y for n in spheres.Nodes]
        vy = [n.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y)
              for n in spheres.Nodes]
        self.summary = {
            "particles_at_end": spheres.NumberOfElements(),
            "wall_conditions": walls.NumberOfConditions(),
            "y_min": min(ys) if ys else None,
            "y_max": max(ys) if ys else None,
            "max_abs_velocity_y": max((abs(v) for v in vy), default=None),
        }
        super().Finalize()


with open("ProjectParametersDEM.json") as _f:
    parameters = KratosMultiphysics.Parameters(_f.read())

analysis = GranularSettling(KratosMultiphysics.Model(), parameters)
n_expected = __N_PARTICLES__
analysis.Run()

summary = dict(analysis.summary)
summary["particles_at_start"] = n_expected
# Read these back from the deck, not from a value baked in here, so that editing
# ProjectParametersDEM.json by hand keeps the summary honest.
summary["dt"] = parameters["MaxTimeStep"].GetDouble()
summary["final_time"] = parameters["FinalTime"].GetDouble()
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
for _k, _v in sorted(summary.items()):
    print("%s=%s" % (_k, _v))

# Kratos DEM deletes particles that leave the bounding box and says nothing --
# an unstable time step therefore looks like a clean, successful run. Refuse to
# exit 0 on a deck that quietly lost its model.
if summary["particles_at_end"] != n_expected:
    raise SystemExit(
        "DEM lost particles: %d of %d remain. The usual cause is MaxTimeStep "
        "above the stability limit; lower dt_safety_factor."
        % (summary["particles_at_end"], n_expected))
if summary["wall_conditions"] == 0:
    raise SystemExit(
        "no wall conditions were read -- check that "
        "<problem_name>DEM_FEM_boundary.mdpa exists next to this script; "
        "a missing DEM mdpa is only an INFO line, never an error.")
print("DEM analysis completed.")
"""

INPUT_PY = (INPUT_PY.replace("__THREADS__", str(omp_threads))
            .replace("__N_PARTICLES__", repr(n_particles)))
with open("input.py", "w") as f:
    f.write(INPUT_PY)

print("Kratos DEM deck written:")
print("  %sDEM.mdpa               %d CylinderParticle2D particles"
      % (problem_name, n_particles))
print("  %sDEM_FEM_boundary.mdpa  %d RigidFace3D3N conditions (%s)"
      % (problem_name, cond_id, ", ".join(walls)))
print("  MaterialsDEM.json          materials + material_relations + table")
print("  ProjectParametersDEM.json  dt=%.3e (%.3g x Rayleigh %.3e), T=%g, %d steps"
      % (dt, dt / dt_rayleigh, dt_rayleigh, T_end, n_steps))
print("  input.py                   run it with: python input.py")
'''
    return header + body


KNOWLEDGE = {
    "dem": {
        "description": "Discrete Element Method via Kratos DEMApplication",
        "application": "DEMApplication (pip install KratosDEMApplication)",
        "workflow": (
            "Kratos DEM uses ProjectParametersDEM.json + .mdpa + MaterialsDEM.json. "
            "The entry point is DEMAnalysisStage(model, parameters).Run(). "
            "Use run_with_generator where the generator creates all input files "
            "and writes input.py as the execution script."
        ),
        "elements": {
            "3D": ["SphericParticle3D (sphere-sphere contact)", "SphericContinuumParticle3D (bonded)"],
            "2D": ["CylinderParticle2D (disk in 2D)", "CylinderContinuumParticle2D"],
            "cluster": ["Cluster3D (rigid body composed of multiple spheres)"],
        },
        "contact_models": {
            "normal": ["DEM_D_Hertz_viscous_Coulomb (recommended)", "DEM_D_Linear_viscous_Coulomb",
                       "DEM_D_Hertz_viscous_Coulomb_JKR (adhesive)", "DEM_D_Hertz_viscous_Coulomb_DMT"],
            "tangential": ["Coulomb friction (built into normal law)"],
            "rolling": ["DEMRollingFrictionModelConstantTorque", "DEMRollingFrictionModelViscousTorque"],
            "bonding": ["DEM_KDEM (parallel bond)", "DEM_parallel_bond", "DEM_Dempack"],
        },
        "solver_types": ["explicit (velocity Verlet, default)", "explicit (symplectic Euler)"],

        # EVIDENCE BASIS for the 'dem' entries below. Every pitfall marked
        # "(Verified by execution 2026-08-07)" was produced by running Kratos
        # 10.4.3 (Release, GCC-15.2, OpenMP) in an environment carrying all 28
        # applications, driving DEMAnalysisStage on a shipped DEMApplication
        # fixture and injecting exactly one fault per run. Entries marked
        # "(Verified from Kratos source ...)" were read off the C++/Python on
        # disk and NOT executed; they are labelled individually.
        #
        # ProjectParametersDEM.json is a FLAT schema and shares almost nothing
        # with the FEM ProjectParameters.json used by StructuralMechanics /
        # FluidDynamics. Working minimum, all at top level: "problem_name",
        # "FinalTime", "MaxTimeStep", "Dimension", "GravityX/Y/Z",
        # "ElementType", "TranslationalIntegrationScheme",
        # "RotationalIntegrationScheme", "BoundingBox*", plus
        # "solver_settings": {"strategy": ..., "material_import_settings": ...}.
        # There is NO "problem_data" block and NO "time_stepping" block.
        "project_parameters_schema": {
            "shape": "flat, top-level keys \u2014 NOT the FEM problem_data/solver_settings layout",
            "problem_name": "top level; the four mdpa filenames are built from it",
            "FinalTime": "top level; end time. Overrides problem_data.end_time if both exist",
            "MaxTimeStep": "top level; the ONLY key that sets dt. Used verbatim",
            "solver_settings.strategy": (
                "python module basename inside DEMApplication: 'sphere_strategy', "
                "'continuum_sphere_strategy', 'verlet_continuum_sphere_strategy', "
                "'ice_continuum_sphere_strategy'. NOT a solver_type string"
            ),
            "solver_settings.material_import_settings.materials_filename": "MaterialsDEM.json",
        },
        "mdpa_files": (
            "DEM reads FOUR separate mdpa files whose names are problem_name "
            "concatenated with a fixed tag and no separator: <problem_name>DEM.mdpa "
            "(spheres), <problem_name>DEM_FEM_boundary.mdpa (walls), "
            "<problem_name>DEM_Clusters.mdpa, <problem_name>DEM_Inlet.mdpa. "
            "solver_settings.model_import_settings.input_filename does NOT name them "
            "and has no effect on which files are read."
        ),
        "materials_schema": (
            "MaterialsDEM.json has three mandatory top-level keys and is NOT the FEM "
            "'properties' schema: 'materials' (list of {material_name, material_id, "
            "Variables}), 'material_relations' (list of {material_names_list, "
            "material_ids_list, Variables}) and 'material_assignation_table' (list of "
            "[submodelpart_name, material_name_or_id]). Per-particle data "
            "(PARTICLE_DENSITY, YOUNG_MODULUS, POISSON_RATIO) goes in 'materials'; "
            "per-CONTACT-PAIR data (STATIC_FRICTION, DYNAMIC_FRICTION, "
            "COEFFICIENT_OF_RESTITUTION, DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME, "
            "DEM_CONTINUUM_CONSTITUTIVE_LAW_NAME, DEM_ROLLING_FRICTION_MODEL_NAME) goes "
            "in 'material_relations'. There is no 'constitutive_law': {'name': ...} key."
        ),
        "pitfalls": [
            "[Input] ProjectParametersDEM.json is a FLAT schema; writing the FEM layout (a 'problem_data' block plus 'solver_settings': {'solver_type': ...}) does not work. DEM looks for solver_settings.strategy, which names a Python module inside DEMApplication ('sphere_strategy'), not a solver_type label. Signal: RuntimeError 'Error: Getting a value that does not exist. entry string : strategy' raised from kratos/sources/kratos_parameters.cpp:426 via DEM_analysis_stage.SetSolverStrategy, before any mesh is read. (Verified by execution 2026-08-07.)",
            "[Input] DEM builds four mdpa filenames by concatenating problem_name with the fixed tags DEM, DEM_FEM_boundary, DEM_Clusters and DEM_Inlet \u2014 so problem_name 'mycase' means the spheres live in 'mycaseDEM.mdpa', not 'mycase.mdpa'. A missing file is NOT an error: the reader prints one INFO line and returns, and the run completes normally on an empty model. Signal: the literal clause 'not found. Continuing.' on stdout, with the logger label and the built filename around it \u2014 the line reads DEM: Input file DEM.mdpa not found. Continuing. \u2014 then a full successful run whose SpheresPart ends with 0 elements and 0 nodes; measured 0/0 against 4/4 for the same case with the file present. (Verified by execution 2026-08-07.)",
            "[Input] The wall file is silently optional in exactly the same way, which is the more dangerous half: particles then fall through the boundary that the user believes exists. Signal: the same literal clause 'not found. Continuing.', this time as DEM: Input file DEM_FEM_boundary.mdpa not found. Continuing., and RigidFacePart holds 0 conditions while SpheresPart still holds its 4 particles \u2014 measured 0 vs 18 conditions against the same case with the wall file present. No exception, exit code 0. (Verified by execution 2026-08-07.)",
            "[Input] solver_settings.model_import_settings.input_filename is inert for DEM. Setting it to the mdpa base name changes nothing, because file paths come from problem_name; its only reader is the restart utility. Signal: a run whose input_filename points at a real, correctly named mdpa still reports 'not found. Continuing.' \u2014 as Input file DEM.mdpa not found. Continuing. \u2014 whenever problem_name disagrees with the file on disk. (Verified from Kratos source 10.4.3, DEM_analysis_stage.py GetInputFilePath/GetProblemNameWithPath \u2014 the inert-key half was not separately executed.)",
            "[Input] MaterialsDEM.json is NOT the FEM materials schema. A file shaped {'properties': [{'model_part_name':..., 'Material': {'Variables':..., 'constitutive_law': {'name':...}}}]} \u2014 the StructuralMechanics layout \u2014 has none of the keys DEM reads. Signal: RuntimeError 'Error: Getting a value that does not exist. entry string : materials' from materials_assignation_utility.py; dropping only the assignation table instead gives the same error, 'Getting a value that does not exist. entry string :', with material_assignation_table as the interpolated key. (Verified by execution 2026-08-07.)",
            "[Input] PARTICLE_FRICTION is not a Kratos variable at all \u2014 it appears in zero files of the installed distribution, including the compiled libraries. Friction is a per-CONTACT-PAIR property named STATIC_FRICTION and DYNAMIC_FRICTION, set in material_relations, not in materials. Signal: RuntimeError 'Error: Value type for \"PARTICLE_FRICTION\" not defined' from read_materials_utility while reading MaterialsDEM.json; the same name passed to KratosGlobals.GetVariable raises ValueError 'Kernel.GetVariable() ERROR: Variable PARTICLE_FRICTION is unknown.'. (Verified by execution 2026-08-07.)",
            "[Numerical] Particle density is PARTICLE_DENSITY. DENSITY is also a valid Kratos variable, so writing it is accepted by the materials reader and then never read \u2014 Properties[PARTICLE_DENSITY] silently returns the inserted default 0.0, particle mass becomes 0, the integrator divides by it, and the resulting non-finite coordinates fail the bounding-box test so every particle is erased. Signal: the run exits 0 and reports ANALYSIS COMPLETED while SpheresPart goes from 4 elements to 0 elements / 0 nodes; probing Properties shows PARTICLE_DENSITY = 0.0 alongside DENSITY = 4000.0. No warning of any kind. (Verified by execution 2026-08-07.)",
            "[Numerical] Omitting YOUNG_MODULUS behaves the same way: the property read inserts 0.0 rather than raising, giving zero contact stiffness and mutually transparent particles. Signal: Properties[YOUNG_MODULUS] reads back exactly 0.0 after a clean Initialize() and the run proceeds to completion with no message. (Verified by execution 2026-08-07.)",
            "[Numerical] MaxTimeStep is the only key that sets dt and it is used verbatim \u2014 DEM performs no stability check and prints no critical-time-step warning at any value. The keys that look like safety nets are dead: DeltaTimeSafetyFactor is stored and never read, AutomaticTimestep is never read at all, and DEM_timestep_safety_factor is not a Kratos key (zero occurrences in the distribution). Signal: MaxTimeStep = 1.0 s on a case whose stable step is 5e-4 prints 'DEM: Total number of time steps expected in the calculation: 0', sets DELTA_TIME to 1.0 and jumps straight to the end time \u2014 no error, no warning. (Verified by execution 2026-08-07.)",
            "[Input] FinalTime, not problem_data.end_time, decides when a DEM run stops, and the DEM default silently overwrites whatever end_time the user wrote. Signal: a deck with 'problem_data': {'end_time': 10.0} and no FinalTime key prints 'DEM: Total number of time steps expected in the calculation: 100' and stops at TIME = 0.05 \u2014 the FinalTime default, 200x short \u2014 with no warning. (Verified by execution 2026-08-07.)",
            "[Input] Every sub model part that carries entities needs a row in material_assignation_table, including the wall part. Omitting the wall row while the wall mdpa is present is a hard error raised deep in the solver, not at materials-reading time. Signal: RuntimeError 'PROPERTIES_ID is not set for SubModelPart' <identifier> '. Make sure the Materials file contains material assignation for this SubModelPart', from explicit_solver_strategy InitializeFEMElements and wrapped at runtime with an 'Error: ' prefix. The two quoted halves are separate literals in libKratosDEMCore.so with an interpolation slot between them, so what appears in the middle is whatever identifies the offending part on that path: measured here as the sub model part NAME (DEM-FEM-Wall_floor, for a wall read from an mdpa) and reported previously as a numeric id (1). Those are runtime values, not literals, which is why neither is quoted here. Both are real renderings of one message — do not treat either as a correction of the other, and do not grep the concatenated sentence, which matches nothing. (Verified by execution 2026-08-07; fixture dem_every_entity_submodelpart_needs_an_assignation_row.)",
            "[Input] Unknown or misspelled top-level keys are rejected by ValidateAndAssignDefaults, and so are unknown keys inside solver_settings \u2014 but validation is flat, so a typo nested inside any other sub-block is silently ignored. Signal: 'MaxTimestep' (lowercase s) raises RuntimeError 'Error: The item with name \"MaxTimestep\" is present in this Parameters but NOT in the default values', followed by a several-hundred-line dump of both the supplied and the default parameter trees; the same happens for an unknown key under solver_settings. (Verified by execution 2026-08-07.)",
            "[API] The contact law name is set per contact PAIR, as DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME inside a material_relations entry \u2014 not as a 'constitutive_law' block on a material. The lookup is an unchecked dictionary get, so a wrong or abbreviated name never names itself in the error. Signal: an abbreviated name such as 'DEM_D_Hertz' produces TypeError \"'NoneType' object is not callable\" raised inside sphere_strategy.ModifySubProperties; the offending string is never echoed. Registered names include DEM_D_Hertz_viscous_Coulomb, DEM_D_Linear_viscous_Coulomb, DEM_D_Hertz_viscous_Coulomb_JKR/_DMT, DEM_KDEM, DEM_parallel_bond, DEM_Dempack. (Verified by execution 2026-08-07.)",
            "[API] Kratos DEM is NOT 3D-only, contrary to a long-standing catalog claim. There is no SphericParticle2D, but genuine 2D particle elements exist under the Cylinder* stem and construct successfully: CylinderParticle2D and CylinderContinuumParticle2D. The right fix for a planar problem is CylinderParticle2D, not a 3D sphere with constrained out-of-plane DOFs. Signal: CreateNewElement('CylinderParticle2D', ...) succeeds on the same model part where CreateNewElement('SphericParticle2D', ...) raises 'The Element \"SphericParticle2D\" is not registered!'; the 2D discontinuum law is spelled DEM_D_Hertz_viscous_Coulomb2D. (Verified by execution 2026-08-07 \u2014 corrects the earlier 'DEM is always 3D internally' wording.)",
            "[API] Wall geometry is built from CONDITIONS, not elements, and only 3D face spellings are registered. Signal: CreateNewCondition('RigidFace3D3N' / 'RigidFace3D4N', ...) constructs, while RigidFace2D2N and RigidEdge3D2N raise 'The Condition \"...\" is not registered!' and list the available DEM conditions. (Verified by execution 2026-08-07.)",
            "[API] RADIUS is a core KratosMultiphysics nodal variable, not a DEMApplication one, and it is the single mandatory per-node input in the spheres mdpa. Signal: KratosMultiphysics.RADIUS resolves while DEMApplication.RADIUS raises AttributeError; a particle whose RADIUS was never written keeps the silent default of zero rather than raising. (Verified by execution 2026-08-07.)",
            "[Output] There is no working VTK path for a DEM SpheresPart on a stock build, and it fails three different ways. (1) DEM particle geometries have no VTK writer at all. Signal: KM.VtkOutput(spheres_part, settings).PrintOutput() raises RuntimeError, built from the two adjacent literals 'Modelpart contains elements or conditions with' and 'geometries for which no VTK-output is implemented!' at kratos/input_output/vtk_output.cpp:451, VtkOutput::WriteCell, so the line you read is: Error: Modelpart contains elements or conditions with geometries for which no VTK-output is implemented! followed by Cell type: 30 — measured for both CylinderParticle2D and SphericParticle3D, which report the same cell type. The Error: prefix, the runtime slot between the two literals and the cell-type suffix are deliberately left OUTSIDE the quotes: they are assembled around the message at runtime, so a quoted string carrying them matches nothing on disk. (2) The core vtk_output_process cannot even get that far, because its Check() runs before DEM has populated the part's ProcessInfo: output_control_type 'step' raises 'Error: STEP not found in process info of SpheresPart.' and 'time' raises the same for TIME, both from OutputController::Check() at kratos/controllers/output_controller.cpp:81. That one is ordering, not absence — DEM_analysis_stage.Initialize does its own work and only then calls AnalysisStage.Initialize; probing the same ProcessInfo after Initialize returns shows TIME present (STEP still absent). (3) post_vtk_option: true raises ModuleNotFoundError('pyevtk') from SetGraphicalOutput, because DEMApplication.dem_vtk_output opens with 'from pyevtk import hl' and pyevtk is neither a Kratos dependency nor shipped with it. What does work is post_gid_option: true, which writes <problem_name>_Post_Files. (Verified by execution 2026-08-07.)",
            # GREP TRAP, and the reason this entry was briefly deleted as a
            # suspected fabrication on 2026-08-07 before being restored the same
            # day. The geometry message above does not exist as one literal
            # anywhere in the distribution. libKratosCore.so holds it as TWO
            # adjacent literals with a runtime slot between them:
            #
            #     b'Modelpart contains elements or conditions with '
            #     b'geometries for which no VTK-output is implemented!'
            #
            # so a fixed-string search for the WHOLE SENTENCE returns 0 files
            # whether you write one space or two between "with" and
            # "geometries". It is not invented: the execution above prints it.
            # On this build (Kratos 10.4.3 at /mnt/kratos-tier2/kv) the slot
            # renders EMPTY, giving one space; a build where it renders
            # non-empty shows an extra token there, which is why the same
            # message has been reported with two.
            #
            # BE PRECISE ABOUT WHAT FAILED HERE, because the honest version is
            # less flattering than the convenient one. The project's own
            # auditor was NOT fooled by this string. audit_quoted_diagnostics
            # falls back to longest_found_prefix when a whole fragment misses,
            # and the leading run "Modelpart contains elements or conditions
            # with" IS a literal, so even the older left-anchored version
            # resolved it and never flagged the entry. What produced the wrong
            # deletion was a hand-run `grep -r -a -F` of the full sentence,
            # returning 0 and being read as proof — going around the guard that
            # exists for exactly this. Grep the interior, or use the auditor.
            #
            # A DIFFERENT and genuinely harder shape sits next to it: every
            # KRATOS_ERROR prepends "Error: " at runtime, so for those the
            # literal in the binary is an INTERIOR run and NO leading-prefix
            # search can reach it -- "Error: STEP not found in process info of
            # SpheresPart." scores 0 at every leading slice while
            # "not found in process info of" is present. Those need a
            # contiguous-window search. PROPERTIES_ID (see the
            # material_assignation_table entry) is split the same way.
        ],
        "guidance": [
            "[Numerical] There is no automatic time step. Size dt by hand from the Rayleigh/Hertz estimate dt ~ 0.34*sqrt(m/(E*pi*R)) and set it in MaxTimeStep; the formula exists in the source (SphericContinuumParticle::Calculate for DELTA_TIME) but is never invoked by any solver path.",
            "[Numerical] The one opt-in automatic-dt path is the automatic_dt_process, added by hand to 'processes'; it prints 'Automatic DT process: Calculated critical time step: <x> seconds.'. It casts unconditionally to SphericContinuumParticle, so pointing it at a plain SphericParticle3D model part dereferences a null pointer. (Source-read, not executed.)",
            "[Numerical] BoundingBoxOption defaults to false. With it true, particles leaving the box are deleted with no message and no counter, and mass is silently not conserved; the check runs only on neighbour-search steps. With it false nothing is ever deleted and escapees integrate forever.",
            "[Numerical] NeighbourSearchFrequency defaults to 50 and SearchTolerance to 0.0, so contacts are rediscovered only every 50 steps with a search radius equal to the particle radius \u2014 fast particles tunnel through each other silently.",
            "[Numerical] For >10k particles: enable MPI via parallel_type: MPI.",
        ]
    },
}

GENERATORS = {
    "dem_2d": _dem_2d_kratos,
}
