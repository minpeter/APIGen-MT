"""Domain-specific hints for API generation by category.

Keys are focus_category names matching BFCL categories.
Empty strings for domains that don't need special rules yet.
"""

DOMAIN_HINTS = {
    "Vehicle Control": """
=== VEHICLE CONTROL DOMAIN RULES ===
When generating queries for Vehicle Control:

1. TRIP FEASIBILITY QUERIES:
   - If user asks "can I make it to [destination]?" or "is the trip feasible?":
     - The distance MUST come from either:
       a) User explicitly states it: "can I make the 380 mile trip to Grand Canyon?"
       b) estimate_distance is called FIRST to compute the actual distance
     - Never use arbitrary distances like 100, 200, 300 unless user specifies
   - Do NOT call estimate_drive_feasibility_by_mileage alone without knowing actual distance

2. NAVIGATION SETUP:
   - When setting navigation destination, include a REAL distance via estimate_distance
   - Example: "Navigate to Grand Canyon (about 600 miles from here) and check if I have enough fuel"
   - expected_tools: [estimate_distance, estimate_drive_feasibility_by_mileage] OR [set_navigation, estimate_distance, estimate_drive_feasibility_by_mileage]

3. FUEL-RELATED QUERIES:
   - When user asks to check fuel level, displayCarStatus(option="fuel")
   - When user asks to add fuel, use fillFuelTank with fuelAmount
   - Ensure requests match tool capabilities (displayCarStatus shows ONE option at a time)

4. DISPLAY TOOLS:
   - displayCarStatus can only show ONE option per call (fuel, battery, doors, climate, etc.)
   - If user asks for multiple statuses, generate multiple tool calls
   - Example: "Check fuel and battery" -> [displayCarStatus(option="fuel"), displayCarStatus(option="battery")]

5. PREREQUISITE TOOLS:
   - Do NOT add prerequisite tools unless user explicitly mentions them
   - Example: "Start the engine" -> startEngine only, NOT pressBrakePedal + startEngine
""",
    "Travel Booking": "",
    "Finance": "",
    "Communication": "",
    "Science": "",
    "Storage": "",
    "Events": "",
    "Posting API": "",
}


def get_domain_hints(focus_category: str) -> str:
    """Get hints for a specific domain, or empty string if none defined."""
    return DOMAIN_HINTS.get(focus_category, "")