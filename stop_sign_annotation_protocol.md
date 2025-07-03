# Stop Sign Annotation Guidelines

**Hello annotators!**  
Your task is to watch dashcam video clips and label how drivers behave when approaching stop signs. This helps improve AI for real-world driving — thank you!

🧠 Use your judgment, stay consistent, and focus only on what you can clearly observe in the video.

---

### 1️⃣ Is a stop sign visible in the clip?
- [ ] Yes  
- [ ] No (if no, skip the rest of the form)

### 2️⃣ Did the driver **fully stop**?
- [ ] Yes – full stop (wheels completely stop)  
- [ ] Rolling stop – slows down but doesn’t fully stop  
- [ ] No – drives through without slowing down

### 3️⃣ What happens **after** the stop sign?
- [ ] Continued straight  
- [ ] Turned left  
- [ ] Turned right  
- [ ] Stopped again shortly after  
- [ ] Unclear (e.g. video ends early or view is blocked)

### 4️⃣ Was the stop sign clearly visible to the driver?
- [ ] Yes – fully visible and obvious  
- [ ] Partially visible – obstructed by trees, vehicles, lighting  
- [ ] No – barely or not visible at all

### 5️⃣ Was there any cross-traffic or pedestrian present at the intersection?
- [ ] Yes – Cross-traffic or pedestrians are visible  
- [ ] No – No cross-traffic or pedestrians are visible  
- [ ] Uncertain – Not clear from the video

### 6️⃣ (Optional) Add a comment if something stood out

Example: "Stop sign was blocked by a truck", "Driver stopped very late", or "Visibility was poor due to fog."


## Key Notes

- Watch the clip more than once if you're unsure.
- Pause to check the moment of stopping.
- Don’t assume — if you can't tell, mark "Unclear."

---

##  Why This Matters
By labeling whether drivers stop (and how), we’re helping train a model to:
- Detect stop signs
- Understand typical and atypical stop behavior
- Enhance autonomous driving decisions

Your careful work will be used to improve road safety — globally. 

---

Thank you for your accuracy and attention! 🙏  
For questions or issues, contact: `yanivm@nexar.com`

---

# Assumptions 
- Only the main driver’s behavior in the video is relevant
- Stop signs must be visually confirmed. I chose it as seperate question to avoid ambiguity. 
- no access to speed, GPS, or other sensor data—only the video
- Driver intent is not inferred beyond raw video (no sound).
- If annotator is not sure or does not see a stop sign - we move forward with the observatiron to avoid "noise" data which can lead to a model bias.
- The annontotors should have clear and straight forward actions to take. Also, we will back-them-up in case of confusion.This is why I added the open field, which will be rarely used in models, and its main goal is allow write ambiguoity details and avoid frustration on the annotator side.

# How These Labels Support Model Training

- Stop sign presence (Q1) helps the fromer model confirm the detection of stop signs in a video and validate the current one to accurate the details.

- Driver stopping behavior (Q2) teaches the model to distinguish proper compliance by the driver.

- Post-stop action (Q3) aids the model in predicting driver intent and planning outcomes.

- **Stop sign visibility** (Q4) helps the model distinguish between cases where the driver fails to stop due to **poor visibility** versus **reckless behavior**.  
Likewise, cross-traffic or pedestrian presence (Q5) supports understanding whether the driver’s behavior was affected by external complications or road conditions.  
These were intentionally separated into two distinct questions (as derived from the annotation section) to avoid ambiguity and help annotators label each factor accurately.  
Later, data scientists may choose to combine these labels into a single feature representing overall **“road distractions.”**

- **Optional comments**, when filled, can:  
  1. Guide model refinement and help identify edge cases.  
  2. Allow annotators to quickly move past unclear cases without overthinking, reducing time spent on edge scenarios while still flagging them for later review.  
  These flagged cases can be reviewed later by the annotation team lead, depending on the current modeling priorities — whether focused on **speed**, **accuracy**, or **regulatory precision**.