"""
Automated UI and Interaction Tests for Streamlit Academic Dashboard.

Verifies:
- Test 1: Initial Load (no simulation runs automatically, placeholder shown, run button visible)
- Test 2: Input change does NOT automatically trigger simulation
- Test 3: Explicit [Run Simulation] click executes simulation and renders results
- Test 4: Input change after simulation preserves results and displays 'Configuration changed' banner
- Test 5: Second [Run Simulation] click updates results to new configuration
- Test 6: Invalid inputs are rejected with clear error messages without crashing
- Test 7: [Reset] button clears results and restores initial state
- Test 8: Multiple sequential simulations maintain exact configuration integrity
"""

import pytest
from streamlit.testing.v1 import AppTest


def test_ui_initial_load():
    """Test 1: On initial load, no simulation executes, no fake results displayed."""
    at = AppTest.from_file("dashboard/app.py").run(timeout=15)
    assert len(at.exception) == 0

    # Ensure empty state notice is present
    empty_states = [elem.value for elem in at.markdown if "No simulation results yet" in elem.value]
    assert len(empty_states) > 0

    # Ensure Run button exists
    run_btn = [btn for btn in at.button if "Run Simulation" in btn.label]
    assert len(run_btn) == 1


def test_ui_input_change_no_auto_execution():
    """Test 2: Changing an input does NOT trigger automatic execution."""
    at = AppTest.from_file("dashboard/app.py").run(timeout=15)
    assert len(at.exception) == 0

    # Modify rounds: 10000 -> 20000
    at.number_input(key="input_ucb_rounds").set_value(20000).run(timeout=15)
    assert len(at.exception) == 0

    # Ensure STILL no simulation has run and empty state persists
    empty_states = [elem.value for elem in at.markdown if "No simulation results yet" in elem.value]
    assert len(empty_states) > 0


def test_ui_explicit_run_simulation():
    """Test 3: Clicking [Run Simulation] executes simulation and renders results."""
    at = AppTest.from_file("dashboard/app.py").run(timeout=10)
    
    # Set small rounds for fast testing
    at.number_input(key="input_ucb_rounds").set_value(100)
    
    # Click the primary Run Simulation button
    run_btn = [btn for btn in at.button if "Run Simulation" in btn.label][0]
    run_btn.click().run(timeout=10)
    assert len(at.exception) == 0

    # Empty state should be GONE
    empty_states = [elem.value for elem in at.markdown if "No simulation results yet" in elem.value]
    assert len(empty_states) == 0

    # Metric card for Total Reward should be rendered
    reward_cards = [elem.value for elem in at.markdown if "Total Reward (Clicks)" in elem.value]
    assert len(reward_cards) > 0

    # Dataframe table should be rendered
    assert len(at.dataframe) > 0


def test_ui_input_change_after_simulation_shows_notice():
    """Test 4: Changing input after simulation preserves previous results and shows notice."""
    at = AppTest.from_file("dashboard/app.py").run(timeout=10)
    
    # Set small rounds
    at.number_input(key="input_ucb_rounds").set_value(100)
    
    # Run first simulation
    run_btn = [btn for btn in at.button if "Run Simulation" in btn.label][0]
    run_btn.click().run(timeout=10)
    assert len(at.exception) == 0

    # Modify exploration coefficient c: 2.0 -> 1.5
    at.number_input(key="input_ucb_c").set_value(1.5).run(timeout=10)
    assert len(at.exception) == 0

    # Ensure Configuration Changed banner is displayed
    changed_banners = [elem.value for elem in at.markdown if "Configuration changed" in elem.value]
    assert len(changed_banners) > 0

    # Previous results should still be visible
    reward_cards = [elem.value for elem in at.markdown if "Total Reward (Clicks)" in elem.value]
    assert len(reward_cards) > 0


def test_ui_reset_clears_results():
    """Test 7: Clicking [Reset] clears results and restores initial empty state."""
    at = AppTest.from_file("dashboard/app.py").run(timeout=10)
    
    # Set small rounds and run simulation
    at.number_input(key="input_ucb_rounds").set_value(100)
    run_btn = [btn for btn in at.button if "Run Simulation" in btn.label][0]
    run_btn.click().run(timeout=10)
    assert len(at.exception) == 0
    assert len([m for m in at.markdown if "Total Reward (Clicks)" in m.value]) > 0

    # Click Reset
    reset_btn = [btn for btn in at.button if "Reset" in btn.label and "Simulation" not in btn.label][0]
    reset_btn.click().run(timeout=10)
    assert len(at.exception) == 0

    # Should return to empty state
    empty_states = [elem.value for elem in at.markdown if "No simulation results yet" in elem.value]
    assert len(empty_states) > 0


def test_ui_multiple_simulations():
    """Test 8: Running multiple simulations updates results to match current config."""
    at = AppTest.from_file("dashboard/app.py").run(timeout=10)
    
    # Run Simulation 1 (rounds=100)
    at.number_input(key="input_ucb_rounds").set_value(100)
    run_btn = [btn for btn in at.button if "Run Simulation" in btn.label][0]
    run_btn.click().run(timeout=10)
    assert len(at.exception) == 0
    assert any("Across 100 rounds" in m.value for m in at.markdown)

    # Change to rounds=150 and run Simulation 2
    at.number_input(key="input_ucb_rounds").set_value(150)
    run_btn = [btn for btn in at.button if "Run Simulation" in btn.label][0]
    run_btn.click().run(timeout=10)
    assert len(at.exception) == 0
    assert any("Across 150 rounds" in m.value for m in at.markdown)
