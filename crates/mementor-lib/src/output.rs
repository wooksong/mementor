use std::io::{Stderr, Stdout, Write};

/// Abstracts stdout/stderr for dependency injection and testability.
pub trait ConsoleOutput<OUT: Write, ERR: Write> {
    fn stdout(&mut self) -> &mut OUT;
    fn stderr(&mut self) -> &mut ERR;
}

/// Real implementation that writes to actual stdout/stderr.
pub struct StdOutput {
    stdout: Stdout,
    stderr: Stderr,
}

impl StdOutput {
    #[must_use]
    pub fn new() -> Self {
        Self {
            stdout: std::io::stdout(),
            stderr: std::io::stderr(),
        }
    }
}

impl Default for StdOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsoleOutput<Stdout, Stderr> for StdOutput {
    fn stdout(&mut self) -> &mut Stdout {
        &mut self.stdout
    }

    fn stderr(&mut self) -> &mut Stderr {
        &mut self.stderr
    }
}

/// Test implementation that captures output into byte buffers.
#[derive(Default)]
pub struct BufferedOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl BufferedOutput {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the captured stdout content as a string.
    pub fn stdout_to_string(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// Returns the captured stderr content as a string.
    pub fn stderr_to_string(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

impl ConsoleOutput<Vec<u8>, Vec<u8>> for BufferedOutput {
    fn stdout(&mut self) -> &mut Vec<u8> {
        &mut self.stdout
    }

    fn stderr(&mut self) -> &mut Vec<u8> {
        &mut self.stderr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffered_output_captures_stdout() {
        let mut output = BufferedOutput::new();
        writeln!(output.stdout(), "hello").unwrap();
        assert_eq!(output.stdout_to_string(), "hello\n");
    }

    #[test]
    fn buffered_output_captures_stderr() {
        let mut output = BufferedOutput::new();
        writeln!(output.stderr(), "error").unwrap();
        assert_eq!(output.stderr_to_string(), "error\n");
    }

    #[test]
    fn buffered_output_starts_empty() {
        let output = BufferedOutput::new();
        assert!(output.stdout_to_string().is_empty());
        assert!(output.stderr_to_string().is_empty());
    }
}
