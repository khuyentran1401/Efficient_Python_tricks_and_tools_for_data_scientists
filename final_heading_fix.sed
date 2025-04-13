# Main title stays as is
/^# Classes$/b

# Convert remaining major section headings to level 2 if they aren't already
s/^# \(Instance Comparison in Python Classes\)$/## \1/

# Leave code comments and command lines as is
/^# [!%]/b
/^# Create /b
/^# Save /b
/^# Using /b
/^# Instance is/b
/^# Access /b
/^# Example /b
/^# Attempting /b
/^# Messy /b
/^# Imagine /b
/^# Serialize /b
/^# Enhanced /b
/^# These /b
/^# If you /b
/^# This will /b
/^# analyze_/b
