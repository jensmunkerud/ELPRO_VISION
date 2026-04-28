import RPi.GPIO as GPIO
import signal
import sys
import time

LIGHT_PINS = [17, 27, 22, 16, 12, 25] #12,25

GPIO.setmode(GPIO.BCM)

for pin in LIGHT_PINS:
	GPIO.setup(pin, GPIO.OUT)
	GPIO.output(pin, GPIO.HIGH)

def cleanup(signum=None, frame=None):
	print("Turning lights off")
	for pin in LIGHT_PINS:
		GPIO.output(pin, GPIO.LOW)
	GPIO.cleanup()
	sys.exit(0)

signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)

while True:
	time.sleep(1)
