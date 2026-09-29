# Real-domain continuation after sub102r3

The follow-up hypothesis was that adding actual speech and music sources would improve the mixture-trained backbone. The two folds grouped related speakers, songs, and parent files; each continuation trained on one side and evaluated on the other. The local fixed-weight objective improved from about 0.1317 to 0.1146 EER, around 1.7 percentage points.

Official public results contradicted the panel signal: **sub106 0.8439324974**, **sub111 0.8394396402**, and **sub112 0.8425539259**, all below sub102r3's 0.8452467831. sub111 used all real-domain rows; sub112 added one more epoch and recovered part of the loss. The team kept sub102r3. These panels were grouped but were still repeatedly used throughout development and did not represent the hidden test distribution.
