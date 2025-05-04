# /home/ubuntu/persistent_watching_listening/decision_logic/simple_logic.py

import time

class SimpleDecisionLogic:
    """Makes decisions based on inputs from watching and listening modules (offline context).
    
    Note: In an offline dataset processing context, the 'cooldown' might be less relevant 
    unless processing sequential data with timestamps. It's kept here for consistency 
    but might need adjustment based on how the dataset is processed.
    """
    def __init__(self, alert_cooldown_seconds=30):
        """Initializes the decision logic.

        Args:
            alert_cooldown_seconds (int): Minimum time in seconds between triggering 
                                          high-priority alerts.
        """
        self.last_alert_time = 0
        self.alert_cooldown_seconds = alert_cooldown_seconds
        # State tracking might be less useful for independent file processing

        # Define action codes
        self.ACTION_NO_ACTION = "NO_ACTION"
        self.ACTION_POTENTIAL_ISSUE = "POTENTIAL_ISSUE" # e.g., fall detected but no keyword
        self.ACTION_ALERT_EMERGENCY = "ALERT_EMERGENCY" # e.g., fall + keyword, or strong keyword

    def process_inputs(self, is_fall_detected, detected_keyword):
        """Processes inputs derived from image/audio file analysis and returns an action code.

        Args:
            is_fall_detected (bool): True if a fall pose was detected in the image.
            detected_keyword (str | None): The keyword detected in the audio file, or None.

        Returns:
            str: An action code (e.g., ACTION_NO_ACTION, ACTION_ALERT_EMERGENCY).
        """
        current_time = time.time() # Used for cooldown, might need timestamp from data if sequential
        action = self.ACTION_NO_ACTION
        trigger_alert = False

        # --- Decision Rules (Remain largely the same) --- 

        # Default English keywords
        help_keywords = ["help", "emergency", "fall"]
        is_help_keyword = detected_keyword is not None and detected_keyword.lower() in help_keywords

        # Rule 1: Fall detected AND help keyword detected = High Priority Emergency
        if is_fall_detected and is_help_keyword:
            print("  Decision Logic: Fall detected AND help keyword detected.") # EN
            trigger_alert = True
        
        # Rule 2: Only Fall detected = Potential Issue
        elif is_fall_detected and not is_help_keyword:
            print("  Decision Logic: Fall detected (no keyword). Potential issue.") # EN
            action = self.ACTION_POTENTIAL_ISSUE 

        # Rule 3: Only Help keyword detected = Potential Emergency (might need confirmation)
        elif not is_fall_detected and is_help_keyword:
            print(f"  Decision Logic: Help keyword '{detected_keyword}' detected (no fall). Potential emergency.") # EN
            # Decide if keyword alone is enough for alert
            strong_keywords = ["emergency"] # Example strong keywords in English
            if detected_keyword.lower() in strong_keywords:
                 trigger_alert = True
            else:
                 action = self.ACTION_POTENTIAL_ISSUE

        # --- Alert Cooldown (Interpret with caution in offline mode) --- 
        if trigger_alert:
            if (current_time - self.last_alert_time) > self.alert_cooldown_seconds:
                print("  Decision Logic: Triggering EMERGENCY ALERT.") # EN
                action = self.ACTION_ALERT_EMERGENCY
                self.last_alert_time = current_time
            else:
                print(f"  Decision Logic: Emergency alert condition met, but within cooldown period. Last alert at {self.last_alert_time:.0f}") # EN
                action = self.ACTION_POTENTIAL_ISSUE 

        # State updates removed as they are less relevant for independent file processing

        return action

# --- Example Usage (Simulating offline processing) ---
if __name__ == '__main__':
    logic = SimpleDecisionLogic(alert_cooldown_seconds=5) # Shorter cooldown for example

    print("--- Processing Item 1: Normal Image, Normal Audio ---") # EN
    action1 = logic.process_inputs(is_fall_detected=False, detected_keyword=None)
    print(f"Action 1: {action1}") # EN

    print("\n--- Processing Item 2: Fall Image, Normal Audio ---") # EN
    action2 = logic.process_inputs(is_fall_detected=True, detected_keyword=None)
    print(f"Action 2: {action2}") # EN

    print("\n--- Processing Item 3: Normal Image, Help Audio ('help') ---") # EN
    action3 = logic.process_inputs(is_fall_detected=False, detected_keyword="help")
    print(f"Action 3: {action3}") # EN

    print("\n--- Processing Item 4: Fall Image, Help Audio ('emergency') ---") # EN
    action4 = logic.process_inputs(is_fall_detected=True, detected_keyword="emergency")
    print(f"Action 4: {action4}") # EN

    print("\n--- Processing Item 5: Fall Image, Help Audio ('emergency') - Immediately after (Cooldown) ---") # EN
    action5 = logic.process_inputs(is_fall_detected=True, detected_keyword="emergency")
    print(f"Action 5: {action5}") # EN

    print("\n--- Processing Item 6: Fall Image, Help Audio ('emergency') - After cooldown ---") # EN
    time.sleep(6)
    action6 = logic.process_inputs(is_fall_detected=True, detected_keyword="emergency")
    print(f"Action 6: {action6}") # EN

    print("\nDecision logic example finished.") # EN

