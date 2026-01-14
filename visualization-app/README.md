# mmWave Vibration Visualization App

Real-time visualization application for mmWave radar spectrograms with vibration status detection.

## Features

- **Real-time Spectrogram Display**: Scrolling heatmap visualization
- **Vibration Status Indicator**: Shows Constant, Increasing, or Decreasing
- **Model-Ready**: Easy integration with classification model when ready

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running from Python
```bash
python main.py
```

### Running the Executable
Double-click `mmwave_visualizer.exe` in the `dist` folder.

## Building the Executable

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name mmwave_visualizer main.py
```

## Model Integration

To connect your model, modify `core/model_connector.py`:

```python
class YourModelConnector(ModelConnector):
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def predict(self, spectrogram_data):
        # Return: 'constant', 'increasing', or 'decreasing'
        return self.model.predict(spectrogram_data)
```
