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

## Wake-word testing

“Hi Milk Dragon” triggered openWakeWord during prototype testing. I also explored bilingual keyword spotting with sherpa-onnx and built a [threshold test script](test_nailong.py) to replay labeled WAV clips and check missed and unintended activations for “Hey Nailong”.

Current work focuses on improving Nailong detection and measuring wake-to-response time and false activations on the robot in continuous use.
