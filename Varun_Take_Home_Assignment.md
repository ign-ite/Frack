Subject: Take-Home Assignment - Real-Time Body Framing Guidance in Python

Hi Varun,

As part of the interview process, we’d like you to complete a take-home assignment. The goal is to build a small real-time computer vision prototype in Python.

## Assignment

Build a Python application that uses a live camera feed to detect a person standing in front of the camera and provide real-time audio guidance to help them position themselves correctly in frame.

## Problem Statement

The app should continuously guide the user so that:

- their full body is visible in the camera frame
- their body is large enough in the frame to be meaningfully assessed
- they are not too far to the left or right in the frame, so the full body remains clearly visible and centered enough for reliable observation

The system should provide audio nudges such as:

- "Please step into frame"
- "Move back"
- "Come closer"
- "Move left"
- "Move right"
- "Hold this position"
- "You’re in frame"

Important requirement:
Even if the user’s full body is technically visible, if they are too far from the camera and appear too small for reliable keypoint assessment, the app should ask them to come closer.

## Scope and Assumptions

- Use Python
- A local webcam-based prototype is sufficient; this does not need to run on a phone
- You may assume a single user in frame
- You may optimize for either front view or side view, but please state your assumption clearly
- You may use any reasonable open-source libraries, pretrained models, or frameworks, but document what you used and why
- This is not a model training exercise; a good prototype using existing tools is completely acceptable

## Expected Behavior

At minimum, the app should handle these scenarios:

1. No person detected
   The app should prompt the user to step into frame.

2. Person is too close
   If the body is cropped or too large in frame, the app should ask the user to move back.

3. Person is fully visible but too far
   If the full body is visible but too small for useful assessment, the app should ask the user to come closer.

4. Person is shifted too far left or right
   The app should ask the user to move left or right so the body is properly positioned.

5. Person is acceptably positioned
   The app should indicate that the position is acceptable.

## Requirements

Your solution should include:

- live camera input
- person detection and/or pose estimation
- logic to determine framing quality
- real-time audio feedback
- reasonable throttling or debouncing of feedback so the app does not repeat instructions every frame
- a clear README explaining setup, how to run the app, the approach taken, assumptions made, and known limitations

## What We’re Looking For

We are primarily evaluating:

- problem solving and decomposition
- practical computer vision judgment
- quality of heuristics and decision logic
- handling of noisy real-time input
- code quality and clarity
- thoughtfulness about user experience, especially the audio guidance loop

## Deliverables

Please submit:

- source code
- README with setup and run instructions
- a short demo video, roughly 2 to 5 minutes, showing the app working in a few scenarios
- a short note describing tradeoffs made, limitations, and what you would improve with more time

## Constraints

- Please keep the scope to roughly 4 to 6 hours of work
- No need for a polished UI
- No backend or cloud deployment is required
- If something is ambiguous, make a reasonable assumption, document it, and proceed

## Evaluation Criteria

We’ll review the submission based on:

- correctness of feedback
- stability in live usage
- code structure and implementation quality
- clarity of explanation
- soundness of tradeoffs

## Submission

Please send back:

- a GitHub repo link or zip file
- the demo video
- brief instructions to run the project locally

We’ll discuss your design choices and tradeoffs in the follow-up interview.

Thanks,
Gokul
