export function getAuthErrorMessage(error) {
  if (!error) {
    return "Something went wrong. Please try again.";
  }

  const message = error.message?.toLowerCase() || "";

  // Login
  if (
    message.includes("invalid login credentials")
  ) {
    return "Email or password is incorrect.";
  }

  // Email already exists
  if (
    message.includes("user already registered")
  ) {
    return "An account with this email already exists.";
  }

  // Invalid email
  if (
    message.includes("invalid email")
  ) {
    return "Please enter a valid email address.";
  }

  // Weak password
  if (
    message.includes("password should be at least")
  ) {
    return "Your password is too weak.";
  }

  // Rate limit
  if (
    message.includes("rate limit") ||
    message.includes("too many requests")
  ) {
    return "Too many attempts. Please wait and try again.";
  }

  // Network
  if (
    message.includes("network") ||
    message.includes("fetch")
  ) {
    return "Unable to connect. Please check your internet connection.";
  }

  // Default
  return "Authentication failed. Please try again.";
}