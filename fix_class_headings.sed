# Keep code comments and command lines as is
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

# Keep main title as level 1
/^# Classes$/b

# Make content sections level 2
s/^# \(Inheritance in Python\)/## \1/
s/^# \(Abstract Classes:.*\)/## \1/
s/^# \(Distinguishing Instance-Level.*\)/## \1/
s/^# \(getattr:.*\)/## \1/
s/^# \(`__call__`:.*\)/## \1/
s/^# \(Instance Comparison.*\)/## \1/
s/^# \(Static method:.*\)/## \1/
s/^# \(Minimize Data Risks.*\)/## \1/
s/^# \(Ensure Data Integrity.*\)/## \1/
s/^# \(`__str__`.*\)/## \1/
s/^# \(Simplify Custom.*\)/## \1/
s/^# \(Optimizing Memory.*\)/## \1/
s/^# \(Improve Code.*\)/## \1/
s/^# \(Embrace the Open.*\)/## \1/
s/^# \(Use Mixins.*\)/## \1/
s/^# \(Embracing Duck.*\)/## \1/
