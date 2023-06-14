import numpy as np
import math
from pathlib import Path
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt
import threading
import time
import sounddevice as sd
import wave
import sys
import wave_solver as solver
import test_wave_solver as test
import engine_thing as engine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create visualisation window
        layout = pg.GraphicsLayoutWidget(border=(100,100,100))
        pg.setConfigOptions(antialias=True)
        self.setCentralWidget(layout)

        # Configure layout
        layout.ci.layout.setColumnStretchFactor(0, 2)
        layout.ci.layout.setColumnStretchFactor(3, 1)

        # Define plots
        self.sim_graph = layout.addPlot(title="Simulation", row=0, col=0, rowspan=2, colspan=3)
        self.mid_graph = layout.addPlot(title="Midpoint Pressure", row=0, col=3, rowspan=1, colspan=1)
        self.valve_graph = layout.addPlot(title="Valve Angle: 0°", row=1, col=3, rowspan=1, colspan=1, aspect_ratio=1, aspect_locked=True, auto_range=False)
        self.sim_curve = self.sim_graph.plot(pen='y')
        self.mid_curve = self.mid_graph.plot(pen='y')
        self.valve_curve = self.valve_graph.plot(pen='y')

        # Initialise arrays and variables
        self.sound_array = np.array([])
        self.x = np.array([])
        self.u = np.array([])
        self.valve_angle = 0
        self.recording_buffer = np.array([])
        self.recording_number = 0
        self.gain = 0.25

        # Configure simulation visualisation
        self.sim_graph.setRange(xRange=[0, 3], yRange=[-3, 3])
        self.sim_graph.setMouseEnabled(x=False, y=False)
        self.sim_graph.setLabel(axis='left', text='u')
        self.sim_graph.setLabel(axis='bottom', text='x')
        self.sim_graph.hideButtons()

        # Configure midpoint visualisation
        self.mid_graph.setRange(xRange=[-10000, 0], yRange=[-2.5, 2.5])
        self.mid_graph.setMouseEnabled(x=False, y=False)
        self.mid_graph.setLabel(axis='left', text='u')
        self.mid_graph.setLabel(axis='bottom', text='t')
        self.mid_graph.hideButtons()

        # Configure valve visualisation
        self.valve_graph.setRange(xRange=[-1, 1], yRange=[-1, 1])
        self.valve_graph.setMouseEnabled(x=False, y=False)
        self.valve_curve.setData([1, -1], [0, 0])
        self.valve_graph.getAxis('bottom').setStyle(showValues=False)
        self.valve_graph.getAxis('left').setStyle(showValues=False)
        self.valve_graph.hideButtons()

        # Set statuses
        self.keep_alive = True
        self.is_recording = False

        # Set up threads
        self._timer = None
        self.simulation = threading.Thread(target=self.simulation_thread, args=())
        self.visualisation = threading.Thread(target=self.visualisation_thread, args=())
        self.audio = threading.Thread(target=self.audio_thread, args=())
        self.simulation.start()
        self.visualisation.start()
        self.audio.start()

    # Replace random values with function calls or don't I can't tell you what to do
    def simulation_thread(self):
        # Refresh simulation until keep alive expires
        I = lambda x: 1E5
        V = 0
        f = 0
        c = 402.317
        U_L = 1E5
        dt = 1/24000
        L = 3
        C = 1.0
        
        def U_0(t):
            frac = (t * 1000 / 24) % 1
            if frac <= 0.25:
                return (1E5 + 69E3)        
            else:
                return (max((69E3*(-frac/0.25 + 2)),(0.0)) + 1E5)
            
        u, u_1, u_2, x, t, c, q, C2, B_p, B1, B_m, U_L, unused, I, V, f = solver.initial_conditioner(I, V, f, c, U_0, U_L, dt, C, L, safety_factor=0.7071, attenuation=True)
        
        while self.keep_alive:
            u, u_1, u_2, x, t, cpu_time = solver.single_timestep(I, V, f, U_0, U_L, dt, u, u_1, u_2, x, t, c, q, C2, B_p, B1, B_m)
            self.x = x
            self.u = (u - I(self.x))*(1.0/np.max(u))
            
            # Define the sound array to be the u of the midpoint
            u_aux = self.gain*np.power(self.u, 2)
            
         
            self.sound_array = np.append(self.sound_array, (u_aux[len(u_aux)//2]))
                
            while len(self.sound_array) >= 3000 :
                time.sleep(dt)
            #time.sleep(max((dt - 0.9*(cpu_time + (time.perf_counter_ns() - t0))/1E9), 0.0))            

    def visualisation_thread(self):
        time.sleep(1/30)

        # Initialise arrays
        sound_vis_array = np.array([])
        sound_vis_x = np.array([])

        # Start graphics
        while self.keep_alive:
            # Update simulation plot
            self.sim_curve.setData(self.x, self.u)

            # Update midpoint plot
            if len(sound_vis_array) == 300:
                sound_vis_array = np.append(sound_vis_array[1:], self.sound_array[-1:])
            else:
                sound_vis_array = np.append(sound_vis_array, self.sound_array[-1:])
                while len(sound_vis_x) < len(sound_vis_array):
                    sound_vis_x = np.append(1000 * -len(sound_vis_array)/30, sound_vis_x)
            self.mid_curve.setData(sound_vis_x[:len(sound_vis_array)], sound_vis_array)

            # Loop every 30th of a second until keep alive expires
            time.sleep(1/30)

    def audio_thread(self):
        time.sleep(0.1)

        # Set up stream
        block_size = 3000
        sample_rate = 12000
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )

        # Start stream
        with stream:
            while self.keep_alive:
                sound_array = self.sound_array

                if len(sound_array) > 10:
                    # Create buffer out of evenly spaced sample of most recent midpoint data
                    if len(sound_array) < block_size:
                        buffer = sound_array
                    else:
                        #Only good if the sound-array is not time dependent
                        buffer = sound_array[np.round(np.linspace(0, len(sound_array) - 1, block_size)).astype(int)]

                    # Ensure that sound array is not played empty
                    if len(sound_array) > 0:
                        # Add to recording buffer if needed
                        if self.is_recording:
                            self.recording_buffer = np.append(self.recording_buffer, buffer).astype('float32')

                        # Clear external buffer
                        self.sound_array = np.array([])
                        
                        # Write to stream
                        stream.write(buffer.astype('float32'))
                    
    def save_audio(self):
        # Scale buffer
        buffer = np.int16(self.recording_buffer * 32767)

        # Find free file name
        filepath = Path("audio_" + str(self.recording_number) + ".wav")
        while filepath.is_file():
            self.recording_number += 1
            filepath = Path("audio_" + str(self.recording_number) + ".wav")

        # Open a new WAV file for writing
        with wave.open(str(filepath), 'w') as file:
            # Set parameters of WAV
            file.setnchannels(1)  # Mono
            file.setsampwidth(2)
            file.setframerate(12000)

            # Write buffer to WAV
            file.writeframes(buffer.tobytes())

        self.recording_number += 1

    def keyPressEvent(self, event):
        # Slowly increase valve angle towards 90 degrees if left arrow is pressed
        if event.key() == Qt.Key_Left:
            if self.valve_angle < 90:
                self.valve_angle += 1
                radians = math.radians(self.valve_angle)
                self.valve_curve.setData([math.cos(radians), -math.cos(radians)], [math.sin(radians), -math.sin(radians)])
                self.valve_graph.setTitle("Valve Angle: " + str(self.valve_angle) + "°")

        # Slowly decrease valve angle towards 0 degrees if right arrow is pressed
        if event.key() == Qt.Key_Right:
            if self.valve_angle > 0:
                self.valve_angle -= 1
                radians = math.radians(self.valve_angle)
                self.valve_curve.setData([math.cos(radians), -math.cos(radians)], [math.sin(radians), -math.sin(radians)])
                self.valve_graph.setTitle("Valve Angle: " + str(self.valve_angle) + "°")

        # Toggle recording if spacebar pressed
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self.is_recording:
                self.is_recording = False
                self.mid_graph.setTitle("Midpoint Pressure")
                # Save if recording was enabled
                saving = threading.Thread(target=self.save_audio, args=())
                saving.start()
            else:
                self.is_recording = True
                self.mid_graph.setTitle("⦿ Midpoint Pressure")

    def closeEvent(self, event):
        # Save any live recording
        if self.is_recording:
            saving = threading.Thread(target=self.save_audio, args=())
            saving.start()
        # End keep alive status
        self.keep_alive = False

        event.accept()
        exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    # Give it a good name or keep this one I think it's pretty good
    main.setWindowTitle("The Tube Simulator 9000")
    main.show()
    sys.exit(app.exec_())
    exit()