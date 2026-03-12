"""
Unit tests for validate_path function.

Tests path validation security features including:
- Path traversal detection
- Null byte injection
- URL protocol blocking
- Extension validation
- Type validation
- Max path length check
"""

import pytest
from odak.tools.file import validate_path
import os
import tempfile


class TestValidatePathBasic:
    """Test basic path validation functionality."""

    def test_valid_simple_path(self):
        """Test that simple valid paths pass validation."""
        result = validate_path("test.txt")
        assert result == os.path.abspath("test.txt")
        assert os.path.basename(result) == "test.txt"

    def test_valid_path_with_extension(self):
        """Test path with common extension passes validation."""
        result = validate_path("data/output/image.png")
        assert "image.png" in result
        assert os.path.isabs(result)

    def test_tilde_expansion(self):
        """Test that tilde (~) is expanded to home directory."""
        result = validate_path("~/test_file.txt")
        assert os.path.expanduser("~") in result
        assert "test_file.txt" in result


class TestValidatePathSecurity:
    """Test security validation features."""

    def test_path_traversal_blocked(self):
        """Test that path traversal with .. is blocked."""
        with pytest.raises(ValueError, match="Path traversal"):
            validate_path("../../etc/passwd")

    def test_path_traversal_in_middle(self):
        """Test that path traversal in middle of path is blocked."""
        with pytest.raises(ValueError, match="Path traversal"):
            validate_path("valid/../../../etc/passwd")

    def test_null_byte_blocked(self):
        """Test that null byte injection is blocked."""
        with pytest.raises(ValueError, match="Null bytes"):
            validate_path("test.txt\x00.exe")

    def test_url_http_blocked(self):
        """Test that HTTP URLs are blocked."""
        with pytest.raises(ValueError, match="URL protocols"):
            validate_path("http://example.com/file.txt")

    def test_url_https_blocked(self):
        """Test that HTTPS URLs are blocked."""
        with pytest.raises(ValueError, match="URL protocols"):
            validate_path("https://example.com/file.txt")

    def test_url_ftp_blocked(self):
        """Test that FTP URLs are blocked."""
        with pytest.raises(ValueError, match="URL protocols"):
            validate_path("ftp://example.com/file.txt")


class TestValidatePathExtensions:
    """Test extension validation functionality."""

    def test_extension_validation_allowed(self):
        """Test that allowed extensions pass validation."""
        result = validate_path("test.png", allowed_extensions=[".png"])
        assert result.endswith(".png")

    def test_extension_validation_blocked(self):
        """Test that disallowed extensions are blocked."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_path("malicious.exe", allowed_extensions=[".txt"])

    def test_extension_case_insensitive(self):
        """Test that extension matching is case-insensitive."""
        result = validate_path("test.PNG", allowed_extensions=[".png"])
        assert ".PNG" in result

    def test_extension_without_dot(self):
        """Test that extensions work without leading dot in spec."""
        result = validate_path("test.txt", allowed_extensions=["txt"])
        assert result.endswith(".txt")

    def test_no_extension_filter(self):
        """Test that files without extension fail when filter requires one."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_path("Makefile", allowed_extensions=[".txt"])

    def test_multiple_allowed_extensions(self):
        """Test that multiple extensions can be validated."""
        result = validate_path("test.jpg", allowed_extensions=[".png", ".jpg", ".jpeg"])
        assert result.endswith(".jpg")


class TestValidatePathTypeValidation:
    """Test type validation."""

    def test_non_string_path_type_error(self):
        """Test that non-string paths raise TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            validate_path(123)

    def test_none_path_type_error(self):
        """Test that None path raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            validate_path(None)

    def test_list_path_type_error(self):
        """Test that list path raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            validate_path(["test.txt"])


class TestValidatePathEdgeCases:
    """Test edge cases and special characters."""

    def test_empty_string_path(self):
        """Test that empty string path is handled."""
        result = validate_path("")
        assert isinstance(result, str)

    def test_spaces_in_filename(self):
        """Test that spaces in filename are preserved."""
        result = validate_path("my file name.txt")
        assert "my file name.txt" in result

    def test_unicode_in_filename(self):
        """Test that unicode characters are allowed."""
        result = validate_path("файл文件.txt")
        assert "файл文件.txt" in result

    def test_special_chars_allowed(self):
        """Test that special characters like hyphens and underscores work."""
        result = validate_path("test-file_2026.png")
        assert "test-file_2026.png" in result


class TestValidatePathLengthLimit:
    """Test path length validation."""

    def test_path_exceeds_max_length(self):
        """Test that paths exceeding 260 characters are rejected."""
        long_name = "a" * 300 + ".txt"
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_path(long_name)

    def test_path_at_max_length(self):
        """Test that paths well below max length are accepted."""
        result = validate_path("a" * 100 + ".txt")
        assert isinstance(result, str)


class TestValidatePathIntegration:
    """Test real-world integration scenarios."""

    def test_valid_image_save_path(self):
        """Test typical image save path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_path = os.path.join(tmpdir, "output.png")
            result = validate_path(safe_path, allowed_extensions=[".png"])
            assert result == safe_path

    def test_valid_directory_path(self):
        """Test directory path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_path(tmpdir + "/")
            assert result == os.path.abspath(tmpdir)

    def test_home_directory_expansion(self):
        """Test full tilde expansion in real paths."""
        home_dir = os.path.expanduser("~")
        result = validate_path("~/test.txt")
        assert result.startswith(home_dir)
        assert "test.txt" in result

    def test_mixed_case_extensions(self):
        """Test mixed case with real image formats."""
        # PNG files can have .PNG, .Png, etc.
        result = validate_path("image.PNG", allowed_extensions=[".png"])
        assert result.endswith(".PNG")

        result = validate_path("image.Png", allowed_extensions=[".png", ".jpg"])
        assert result.endswith(".Png")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
