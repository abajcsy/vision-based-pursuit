from pyPS4Controller.controller import Controller
import time
import numpy as np

class Joypad(Controller):
    def __init__(self, **kwargs):
        Controller.__init__(self, interface="/dev/input/js0", connecting_using_ds4drv=False, **kwargs)
        self.time_last_joy = time.time()
        self.joypad_timeout = 10
        self.forward_value_normalized = 0
        self.angular_value_normalized = 0
        
    def on_L3_up(self, value):
        self.time_last_joy = time.time()
        command = -value
        command = np.max(command,0)
        self.forward_value_normalized = command / 32767
        print("forward_value norm: {}".format(self.forward_value_normalized))

    def on_L3_down(self, value):
        self.time_last_joy = time.time()
        self.forward_value_normalized = -value / 32767
        print("forward_value norm: {}".format(self.forward_value_normalized))
        #print("on_L3_down: {}".format(value))

    def on_L3_left(self, value):
        #command = -value
        #command = np.max(command,0)
        self.time_last_joy = time.time()
        self.angular_value_normalized = -value / 32767
        print("angular_value norm: {}".format(self.forward_value_normalized))

    def on_L3_right(self, value):
        #print("on_L3_right: {}".format(value))
        self.time_last_joy = time.time()
        self.angular_value_normalized = -value / 32767
        print("angular_value norm: {}".format(self.forward_value_normalized))

    def on_L3_y_at_rest(self):
        """L3 joystick is at rest after the joystick was moved and let go off"""
        pass
        #print("on_L3_y_at_rest")

    def on_L3_x_at_rest(self):
        """L3 joystick is at rest after the joystick was moved and let go off"""
        pass
        #print("on_L3_x_at_rest")

    def on_L3_press(self):
        """L3 joystick is clicked. This event is only detected when connecting without ds4drv"""
        pass
        #print("on_L3_press")

    def on_L3_release(self):
        """L3 joystick is released after the click. This event is only detected when connecting without ds4drv"""
        pass
        #print("on_L3_release")