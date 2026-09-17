"""
MCP tools for cross-solver benchmarking.

Runs the same problem on all available backends and compares results.
"""

import asyncio
import json
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from core.registry import available_backends, get_backend
from core.post_processing import post_process_file, compare_results

from core.output_paths import output_dir as _output_dir  # noqa: E402
_BENCHMARK_DIR = _output_dir("benchmark_results")


def register_benchmark_tools(mcp: FastMCP):

    @mcp.tool()
    async def benchmark_problem(physics: str, variant: str = "2d",
                                params: str = "{}",
                                solvers: str = "") -> str:
        """Run the same physics problem on all available backends and compare.

        This is the cross-solver validation tool — runs identical problems on
        multiple FEM codes and verifies they produce the same answer.

        Args:
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'heat')
            variant: Template variant (e.g. '2d', '3d')
            params: JSON parameters to override defaults
            solvers: Comma-separated solver names to use (empty = all available)

        Returns:
            Comparison table with DOFs, solve times, field statistics, and
            agreement metrics across solvers.
        """
        import json as _json
        try:
            param_dict = _json.loads(params)
        except _json.JSONDecodeError as e:
            return f"Invalid params JSON: {e}"

        # Determine which backends to use
        if solvers:
            backend_names = [s.strip() for s in solvers.split(",")]
            backends = []
            for name in backend_names:
                b = get_backend(name)
                if b:
                    status, msg = b.check_availability()
                    if status.value == "available":
                        backends.append(b)
                    else:
                        pass  # skip unavailable
        else:
            backends = available_backends()

        # Filter to backends that support this physics+variant
        eligible = []
        for b in backends:
            supported = {p.name: p for p in b.supported_physics()}
            if physics in supported:
                p = supported[physics]
                if variant in p.template_variants:
                    eligible.append(b)

        if not eligible:
            return (f"No available backend supports {physics}/{variant}. "
                    f"Available backends: {[b.name() for b in backends]}")

        # Run on each backend
        _BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        bench_dir = _BENCHMARK_DIR / f"{physics}_{variant}_{ts}"
        bench_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for backend in eligible:
            name = backend.name()
            work_dir = bench_dir / name
            work_dir.mkdir(parents=True, exist_ok=True)

            try:
                content = backend.generate_input(physics, variant, param_dict)
                errors = backend.validate_input(content)
                if errors:
                    results.append({
                        "solver": name,
                        "status": "validation_failed",
                        "errors": errors,
                    })
                    continue

                job = await backend.run(content, work_dir, np=1, timeout=None)

                entry = {
                    "solver": name,
                    "display_name": backend.display_name(),
                    "status": job.status,
                    "elapsed_s": round(job.elapsed, 3) if job.elapsed else None,
                }

                if job.status == "completed":
                    result_files = backend.get_result_files(job)
                    vtu_files = [f for f in result_files if f.suffix == ".vtu"]

                    if vtu_files:
                        pp = post_process_file(vtu_files[0], plot_dir=work_dir)
                        entry["mesh"] = pp.mesh.to_dict()
                        entry["fields"] = [f.to_dict() for f in pp.fields]
                        entry["qa_warnings"] = pp.qa_warnings
                        entry["plots"] = pp.plots
                    else:
                        entry["output_files"] = [f.name for f in result_files]
                elif job.error:
                    entry["error"] = job.error[:500]

                results.append(entry)

            except Exception as e:
                results.append({
                    "solver": name,
                    "status": "error",
                    "error": str(e),
                })

        # Cross-solver comparison
        vtu_paths = []
        for r in results:
            if r["status"] == "completed":
                solver_dir = bench_dir / r["solver"]
                vtus = list(solver_dir.glob("*.vtu"))
                if vtus:
                    vtu_paths.append(vtus[0])

        comparison = {}
        if len(vtu_paths) >= 2:
            comparison = compare_results(vtu_paths)

        # Build output
        output = {
            "benchmark": f"{physics}/{variant}",
            "params": param_dict,
            "n_solvers": len(results),
            "results": results,
            "cross_solver_comparison": comparison,
            "output_dir": str(bench_dir),
        }

        # Save benchmark JSON
        (bench_dir / "benchmark.json").write_text(
            json.dumps(output, indent=2, default=str)
        )

        # Format summary table
        lines = [f"## Benchmark: {physics}/{variant}\n"]
        lines.append("| Solver | Status | DOFs | max(field) | Time |")
        lines.append("|--------|--------|------|------------|------|")
        for r in results:
            status = r["status"]
            dofs = "-"
            field_max = "-"
            elapsed = r.get("elapsed_s", "-")
            if "mesh" in r:
                dofs = r["mesh"]["n_points"]
            if "fields" in r and r["fields"]:
                field_max = f"{r['fields'][0]['max']:.6e}"
            lines.append(f"| {r.get('display_name', r['solver'])} | {status} | {dofs} | {field_max} | {elapsed}s |")

        if comparison.get("agreement"):
            ag = comparison["agreement"]
            lines.append(f"\n**Cross-solver agreement:** {ag['agreement_pct']:.1f}%")
            lines.append(f"**Max relative deviation:** {ag['max_relative_deviation']:.2e}")

        lines.append(f"\nResults saved to: {bench_dir}")

        return "\n".join(lines)

    @mcp.tool()
    async def benchmark_poisson_cross_solver() -> str:
        """Quick benchmark: Poisson on unit square across all available solvers.

        Runs -Δu = 1 on [0,1]², u=0 on ∂Ω with ~1000 DOFs on each solver.
        Expected max(u) ≈ 0.0737 (analytical reference).
        """
        return await benchmark_problem("poisson", "2d", "{}")
