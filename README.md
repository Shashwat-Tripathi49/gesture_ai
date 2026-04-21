# Gesture AI - My Python playground 🐍👋

Hey everyone! Thanks for checking out this little project of mine. 

I've been learning a lot of Python lately, and honestly, the best way for me to actually understand the syntax and logic has been just diving headfirst into random visual projects that sound fun. I wanted to build something that feels kind of like magic, so I threw together this real-time gesture tracking app using Python, OpenCV, and Google's MediaPipe.

### What it actually does
Instead of just tracking a pointing finger or drawing static lines, I wrote a custom physics simulation that connects a flexible, elastic "string" or ribbon to *every single finger* you hold up to the webcam. If you wave your hands, the strings physically whip and trail behind your fingertips in real-time. It tracks all 10 fingers and calculates how the strings should bend using inverse kinematics!

### How to run it locally
If you want to play around with it and tweak the physics yourself:
1. Clone this repo to your machine.
2. Double-click the `run_app.bat` script (I made this so it automatically handles setting up a safe virtual environment and installing the `requirements.txt` for you).
3. The camera window will pop up. Move your hands in front of the lens and watch the strings follow. Press `Q` while clicked on the window to shut it down.

### Tech Stack
- **Python** (Still getting the hang of it, but I'm loving it so far!)
- **OpenCV** for rendering the graphics and pulling my raw webcam feed.
- **MediaPipe Tasks API** for the heavy machine learning models that find hands in the video.
- A whole bunch of geometry/math for the flexible string physics simulation :)

Feel free to fork it, mess around with the string colors, or suggest any cool features I could add as I keep learning!
