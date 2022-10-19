import sys
import traceback


def printerr(msg):
    import sys
    print(msg, file=sys.stderr)


def printerr_bp(msg):
    print(f' - {msg}', file=sys.stderr)


def run(code, task):
    try:
        code()
    except Exception as e:
        print(f"{task}: {type(e).__name__}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

progress_print_out = sys.stdout
