# First, mark code comments and command lines
/^# !.*$/b
/^# %.*/b
/^# Create /b
/^# Save /b
/^# Using /b
/^# Instance /b
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

# Then update main content headings
s/^## Classes$/# Classes/
s/^## \([^!#%][^#]*\)$/# \1/
