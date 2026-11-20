# Robot wake-word detection

**Prototype in progress.** This project explores how a robot can wake on a spoken phrase and route follow-up voice events according to its interaction state. I prototyped two detector paths, openWakeWord with custom ONNX models and sherpa-onnx with a bilingual streaming keyword spotter.

## Try the event simulation

Requires Python 3. Run from this directory.

```bash
python voice_node.py --simulate --debug-all-keywords
```

Enter the following commands one at a time.

```text
NAILONG
OKAY
:state EXECUTING
STOP
:quit
```

The node emits `WAKE`, `CONFIRM`, and `STOP_REQUEST` as JSON. The simulation exercises event handling with typed keywords. A stop request is an application event. The robot's physical emergency stop is independent.

## Offline evaluation

I replayed the same WAV clips through both detectors, then ran a separate focused threshold test for “Hey Nailong”. The [threshold script](test_nailong.py) evaluates a supplied openWakeWord ONNX model against labeled 16 kHz mono WAV files.

| Evaluation | openWakeWord | sherpa-onnx |
| --- | ---: | ---: |
| “Hi Milk Dragon”, 4 clips across two pilot batches | 2/4 at threshold 0.35 | 1/4 with the prototype keyword configuration |
| “Hey Nailong”, 4 clips across the same pilot batches | 0/4 at threshold 0.35 | 0/4 |
| Background clips in the pilot batches | 0/4 triggered | 0/4 triggered |

In a separate 20-clip Nailong test, openWakeWord detected **0/10** positives at **0.20** and **4/10** at **0.02**. Neither setting triggered on the 10 negative clips. These small offline trials helped identify missed detections as the immediate issue. Live response time and false activations during continuous operation still need to be measured.

The next experiment will fix the microphone and distance, compare “奶龙” with “Hey Nailong”, and test the chosen configuration on fresh recordings and continuous background speech.
