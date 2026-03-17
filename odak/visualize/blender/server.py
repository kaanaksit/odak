from libblend import *
import bpy
import queue
import threading
import os
import socket
import traceback
import sys
import ast
import inspect

sys.path.append("%s" % os.path.dirname(os.path.realpath(__file__)))

execution_queue = queue.Queue(maxsize=100)  # Prevent resource exhaustion

ALLOWED_FUNCTIONS = {
    "import_ply",
    "prepare",
    "delete_the_cube",
    "render",
    "cylinder_between",
    "quit",
    "delete_object",
}


class whisper2blender(threading.Thread):
    def __init__(self, port=8082, host="127.0.0.1", path_max=4096):
        threading.Thread.__init__(self)
        self.port = port
        self.host = host
        self.path_max = path_max

    def run(self):
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((self.host, self.port))
        serversocket.listen(1)
        print("Listening on %s:%s" % (self.host, self.port))
        while True:
            connection, address = serversocket.accept()
            buf = connection.recv(self.path_max)
            for cmd in buf.split(b"\x00"):
                if cmd:
                    try:
                        cmd_str = cmd.decode()
                        validate_and_queue_command(cmd_str)
                    except Exception as e:
                        print("Command validation failed:", str(e))
                        traceback.print_exc()


def run_in_main_thread(function_tuple):
    """Queue a function tuple for execution in the main thread."""
    try:
        execution_queue.put(function_tuple)
    except queue.Full:
        print("Command queue full, rejecting command")


def validate_and_queue_command(cmd_str):
    """Validate and queue a command string for safe execution."""
    for func_name in ALLOWED_FUNCTIONS:
        prefix = f"{func_name}("
        if cmd_str.strip().startswith(prefix):
            run_in_main_thread((func_name, cmd_str))
            return
    raise ValueError(
        "Unauthorized function call. Only whitelisted functions are allowed."
    )


def parse_function_call_safely(cmd_str, expected_func_name):
    """
    Safely parse a function call string using AST.

    Security: Uses ast.literal_eval for argument parsing to prevent code injection.
    Only allows literals (numbers, strings, tuples, lists, dicts, booleans, None).

    Parameters
    ----------
    cmd_str : str
        Command string to parse (e.g., "render(123)" or "delete_object('cube')")
    expected_func_name : str
        Expected function name for verification

    Returns
    -------
    tuple
        (func_name, args, kwargs) if successful, (None, [], {}) if failed

    Raises
    ------
    ValueError
        If function name mismatch or unsafe constructs detected
    """
    try:
        # Parse the command string as an AST expression
        tree = ast.parse(cmd_str.strip(), mode="eval")
        call_node = tree.body

        # Ensure it's a function call
        if not isinstance(call_node, ast.Call):
            return None, [], {}

        # Ensure the function is called by name (not an attribute or complex expression)
        if not isinstance(call_node.func, ast.Name):
            return None, [], {}

        func_name = call_node.func.id

        # Verify function name matches expected (prevents name spoofing)
        if func_name != expected_func_name:
            raise ValueError(
                f"Function name mismatch: expected '{expected_func_name}', got '{func_name}'"
            )

        # Safely evaluate arguments using literal_eval (only allows literals)
        args = []
        for arg_node in call_node.args:
            try:
                # Convert AST node back to source and safely evaluate
                arg_source = ast.unparse(arg_node)
                arg_value = ast.literal_eval(arg_source)
                args.append(arg_value)
            except (ValueError, SyntaxError) as e:
                # Reject complex expressions (function calls, attribute access, etc.)
                raise ValueError(
                    f"Unsafe argument detected. Only literals allowed: {str(e)}"
                )

        # Safely evaluate keyword arguments
        kwargs = {}
        for kw_node in call_node.keywords:
            if not isinstance(kw_node.arg, str):
                raise ValueError("Invalid keyword argument name")
            try:
                kw_source = ast.unparse(kw_node.value)
                kw_value = ast.literal_eval(kw_source)
                kwargs[kw_node.arg] = kw_value
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Unsafe keyword argument detected. Only literals allowed: {str(e)}"
                )

        return func_name, args, kwargs

    except SyntaxError as e:
        print(f"Syntax error in command: {e}")
        return None, [], {}
    except ValueError as e:
        print(f"Security validation failed: {e}")
        return None, [], {}


def execute_queued_functions():
    """
    Execute queued functions safely without using exec().

    Security: Uses direct function calls with parsed arguments instead of exec().
    This prevents arbitrary code execution vulnerabilities.
    """
    while not execution_queue.empty():
        function_tuple = execution_queue.get()
        if function_tuple:
            func_name, cmd_str = function_tuple

            # Verify function is in whitelist
            if func_name not in ALLOWED_FUNCTIONS:
                print(f"Security error: Function '{func_name}' not in whitelist")
                continue

            # Verify function exists in globals
            if func_name not in globals():
                print(f"Error: Function '{func_name}' not found")
                continue

            # Safely parse the function call
            parsed_name, args, kwargs = parse_function_call_safely(cmd_str, func_name)

            # Verify parsing succeeded and name matches
            if parsed_name is None or parsed_name != func_name:
                print(f"Failed to safely parse command: {cmd_str[:50]}...")
                continue

            # Get the function reference
            func = globals()[func_name]

            # Verify it's actually callable
            if not callable(func):
                print(f"Error: '{func_name}' is not callable")
                continue

            # Execute the function safely with parsed arguments
            try:
                func(*args, **kwargs)
            except TypeError as e:
                print(f"Function call failed (wrong arguments): {e}")
            except Exception as e:
                print(f"Function execution failed: {e}")
                traceback.print_exc()

    return 0.1


def main():
    messenger = whisper2blender()
    messenger.start()
    bpy.app.timers.register(execute_queued_functions)
    return True


if __name__ == "__main__":
    main()
