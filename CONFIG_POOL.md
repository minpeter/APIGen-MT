# Config Pool — Diverse API State Generation

## What it is

`src/config_pool.py` provides domain-specific configuration pools that randomize the initial API state for each generated datapoint, replacing the previous single hardcoded config (`FULL_INITIAL_CONFIGS` in `tool_manager.py`).

## Why

Before: every datapoint started from the identical API state (e.g., vehicle always `stopped`, fuel `15.0`, destination `Grand Canyon`). This caused argument diversity problems — `lockDoors` always got the same args, `startEngine` always got `START`, 84% of queries were "road trips".

After: each datapoint draws a random config per domain, producing varied starting states (parked/driving/cold morning for vehicles, different trading accounts, different ticket queues, etc.).

## Domains covered

| Domain | Configs | Example variations |
|---|---|---|
| Vehicle Control | 10 | parked, driving, cold morning, hill, low fuel |
| Trading Bot | 5 | admin, retail, fund manager, penny stock, day trader |
| Ticket API | 5 | support queue, admin, HR, ops, empty new agent |
| Travel Booking | 5 | standard, business, luxury, budget, corporate |
| Posting API | 5 | tech blogger, foodie, traveler, newcomer, sports fan |
| Message API | 5 | standard, dev team, classroom, family, startup |
| Filesystem | 5 | standard, code project, photos, server, research |
| Math API | 1 | (stateless, no variation needed) |

Plus **52 personas** and **50 cities** for diverse user query seeds.

## Functions

### `generate_random_config(seed=None)` — API state seed

**Defined:** `src/config_pool.py:878`

Picks one variation per domain using `random.Random(seed)` and combines them into a full config dict matching the `FULL_INITIAL_CONFIGS` structure:

```python
config = {
    "GorillaFileSystem":    rng.choice(FILESYSTEM_CONFIGS),
    "MathAPI":              {"numbers": [275.5, ...]},  # fixed
    "MessageAPI":           rng.choice(MESSAGE_CONFIGS),
    "PostingAPI":           rng.choice(POSTING_CONFIGS),
    "TicketAPI":            rng.choice(TICKET_CONFIGS),
    "TradingBot":           rng.choice(TRADING_CONFIGS),
    "TravelAPI":            rng.choice(TRAVEL_CONFIGS),
    "VehicleControlAPI":    rng.choice(VEHICLE_CONFIGS),
}
```

**Called from:** `src/tool_manager.py:772` in `reset_python_tool_instances()`:

```python
if self.use_config_pool:
    config = generate_random_config()
else:
    config = FULL_INITIAL_CONFIGS
self.python_tool_instances = create_python_tool_instances(config)
```

Which is triggered by `tool_manager.initialize_api_state()` at the start of each datapoint generation.

### `generate_query_seed(rng=None)` — Persona/city seed

**Defined:** `src/config_pool.py:868`

Picks a random persona and city to diversify user queries and tool arguments:

```python
return {"persona": rng.choice(PERSONAS), "city": rng.choice(CITIES)}
```

**Called from:** `src/apigen_step_by_step.py:627` in `generate_datapoint()`:

```python
query_seed = generate_query_seed()
# seed flows to Stage 1 (query generation) and Stage 2 (argument generation)
query_result = self._stage1_generate_query(focus_category, ..., query_seed)
trajectory = self._stage2_generate_tools(query_result, ..., query_seed)
```

The seed is used in:
- **`generate_user_query()`** (`apigen_step_by_step.py:327`) — persona name, city, and occupation injected into the query generation prompt
- **`_generate_tool_arguments()`** (`apigen_step_by_step.py:1031`) — user context (name, city, occupation) injected into argument generation prompt

## CLI usage

```bash
# With config pool (default)
python src/generate_step_by_step.py --config-pool ...

# Without (fallback to original single config)
python src/generate_step_by_step.py --no-config-pool ...
```

## How it works (per datapoint)

1. `ToolManager.__init__(use_config_pool=True)` sets the flag
2. `generate_datapoint()` calls `generate_query_seed()` for persona/city
3. `tool_manager.initialize_api_state()` → `reset_python_tool_instances()` → `generate_random_config()` picks one variation per domain
4. All 8 tool classes instantiated with the randomized config
5. `filter_api_state()` strips unused domains from the saved datapoint (only domains whose tools were called are retained)

## Verified working

Test generation confirmed config pool produces different states per datapoint:

| Field | Config Pool | Original |
|---|---|---|
| engineState | `running` | `stopped` |
| fuelLevel | `30.0` | `15.0` |
| batteryVoltage | `12.4` | `12.8` |
| destination | `None` | `Grand Canyon` |
| remainingUnlockedDoors | `4` | `0` |
| doorStatus | all `unlocked` | all `locked` |
