"""openPASO core: server instructions, attestation, and shared helpers."""
from core.env_compat import install_aliases as _install_aliases

# The project's variables answer to both their old (OASIS_*) and new
# (OPENPASO_*) names from the moment anything in the package is imported.
_install_aliases()
