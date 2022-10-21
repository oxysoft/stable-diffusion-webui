from core.installing import is_installed, run_pip


class NgrokPlugin:
    def install(self, args):
        if not is_installed("pyngrok"):
            run_pip("install pyngrok", "ngrok")