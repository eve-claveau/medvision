# medvision
## Inspiration

Medical video feeds such as ultrasound, echocardiography, and laparoscopy are constantly in motion due to camera movement and natural anatomical motion. While many collaboration tools allow annotations, these annotations are usually static and quickly become misaligned, reducing their usefulness in real clinical or educational settings. We were inspired by HoloRay’s mission to improve healthcare collaboration and learning, and wanted to explore how motion-tracked annotations could make medical video interpretation clearer, more intuitive, and more reliable.

## What it does

Medvision allows users to place annotations on a medical video feed and keeps those annotations anchored to the same visual anatomy as the video moves. Instead of annotations “floating” or drifting when the camera or anatomy moves, the annotation follows the motion of the underlying structure, demonstrating how motion-tracked annotation can improve clarity during live or recorded medical procedures.
How we built it

We built MedVision with:

    Backend / Tracking: Python with OpenCV (cv2) for motion tracking, feature detection, and optical flow experiments
    Frontend / UI: HTML, CSS, and JavaScript to display the video, render annotation overlays, and visualize tracking states

## Challenges we ran into

One of our biggest challenges was designing tracking that worked well with user-placed annotations. Tracking arbitrary points chosen by a user is significantly harder than tracking predefined objects.

We also struggled with precision and stability, especially when dealing with jitter, drift, occlusion, and moments when the annotated region leaves the frame. Finally, integrating the Python tracking logic with the frontend visualization was challenging, particularly ensuring smooth, real-time updates that aligned with video frame rates.
## Accomplishments that we're proud of

    Tackling a technically difficult problem as a team of beginners
    Successfully collaborating across frontend and backend roles
    Designing a UI that clearly demonstrates tracking accuracy, stability, and robustness
    Completing our first hackathon project while learning new tools under time pressure

## What we learned

Through this project, we learned about:

    Motion tracking and computer vision concepts in Python
    OpenCV feature tracking and optical flow techniques
    Team collaboration and problem-solving in a hackathon environment

## What's next for MedVision

Next, we would like to: support multiple annotations at once, improve tracking robustness using more advanced models, and enable collaborative/multi-user annotation.
