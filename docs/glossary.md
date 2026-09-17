# Words used here

Every word you meet on this site, in one line.

| Word | Meaning in one line |
|---|---|
| **solver** | a program that computes the physics |
| **backend** | one such program, as openPASO sees it |
| **finite element method** | cutting an object into small pieces and solving piece by piece |
| **mesh** | the set of small pieces |
| **element** | one piece of the mesh |
| **boundary condition** | what you fix at the edges: a temperature, a force, a fixed wall |
| **refine the mesh** | use more and smaller pieces, for a more accurate answer |
| **mesh independence** | the answer stops changing when you refine further — a good sign |
| **convergence rate / order** | how fast the error shrinks as you refine |
| **coupling** | two solvers working on one problem, exchanging values at their shared boundary |
| **verification** | checking that the numbers are computed correctly |
| **validation** | checking that the model matches the real world — **not** done here |
| **MCP** | the standard plug that connects tools to AI apps |
| **git** | the tool that downloads this project's files (`git clone`) |
| **pip** | the tool that installs Python packages |
| **compiler** | a program that turns source code into something your machine can run. Some packages need one |
| **venv** | a private Python folder for one project's packages |
| **`pip install -e .`** | install the project in this folder (`.`) so your edits take effect straight away (`-e`) |
| **PYTHONPATH** | tells Python which folder to find openPASO's code in |
| **VIRTUAL_ENV** | tells programs which private Python folder to use |
| **PYVISTA_OFF_SCREEN** | draws pictures without opening a window, so it works on a machine with no screen |
| **langgraph** | the library that lets an AI model use tools in a loop; only Option B needs it |
| **Gmsh** | the program openPASO uses to build meshes |
| **preCICE** | a separate library for coupling two solvers; an alternative to openPASO's own `couple` |
| **wheel** | a ready-built Python package that `pip` downloads instead of compiling |
| **YAML**, **XML** | two other text formats for settings; some solvers use these instead of JSON |
| **MPI** / **mpi4py** | the standard way programs split work across many processors |
| **ldd** | a command that prints which system libraries your machine has |
| **DSMC** | a particle method for gas so thin that the usual flow equations stop working |
| **JSON** | a text format for settings. Every bracket, quote and comma must match, or the whole file is ignored |
| **binary** | a program you can run, already compiled — you do not build it yourself |
| **interpreter** | the `python` program itself; several can be installed side by side |
| **prompt** (terminal) | the text your terminal shows before you type, such as `$`. Not the same as the question you ask an AI |
| **fork** | a copy of a project developed separately |
| **conda** | another tool for private Python folders, like venv. FEniCSx needs it |
| **agent** | an AI model that can use tools by itself, not only write text |
| **server** | the background program the AI app talks to; openPASO is one |
| **environment variable** | a setting your terminal passes to a program. `export NAME=value` sets one, and it is forgotten when you close the terminal |
| **API key** | a password that lets a program use a paid AI service |
| **OpenRouter** | a service giving one key access to many AI models |
| **tool support** | whether a model is able to call tools. Not every model is |
| **Poisson equation** | a standard textbook problem used to check that a solver works |
| **unit square** | the square from 0 to 1 in both directions, the usual test shape |
| **participant** | one side of a coupled problem: a small script for one solver |
| **interface** | the boundary two coupled halves share |
| **manufactured solution** | a problem built backwards from a known answer, so the error can be measured |
