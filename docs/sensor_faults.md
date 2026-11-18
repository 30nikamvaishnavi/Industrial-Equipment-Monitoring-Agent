# Sensor Faults & Instrumentation Maintenance Manual

## Overview
Reliable predictive maintenance depends on accurate field instrumentation. Sensor faults can mask real mechanical degradation or trigger false shutdown alarms if not distinguished from physical machine faults.

## Typical Symptoms
- **Flat-line (Frozen) Readings**: Sensor output remains completely constant with zero variance over consecutive hours, or drops abruptly to 0.0.
- **Signal Dropouts / Open Circuit**: Instantaneous drop to rail minimum or NaN/zero values while correlated sensors (e.g., RPM or temperature) remain normal.
- **Wiring Intermittent Spikes**: Single-sample extreme outliers inconsistent with physical thermal or mechanical inertia.

## Likely Causes
1. **Loose Field Wiring or Terminal Corrosion**: Vibration-induced terminal screw loosening or moisture ingress in junction boxes.
2. **Transducer / Thermocouple Failure**: Open RTD element, damaged piezoelectric accelerometer cable, or clogged pressure impulse tube.
3. **PLC / DAQ Input Module Freeze**: Stale buffer register or analog-to-digital converter latch-up.
4. **Power Supply Brownout**: Insufficient loop excitation voltage (24 VDC / 4-20 mA loop interruption).

## Recommended Actions
- **Immediate**: Cross-check suspect channel against redundant sensors and inspect PLC/SCADA communication diagnostics; verify 4-20 mA loop current.
- **Short-Term**: Inspect armored sensor cable, M12 connectors, and junction box terminals for moisture or strain damage.
- **Corrective**: Reseat or replace faulty transducer, clean oxidized terminal blocks, recalibrate transmitter zero/span against reference standard, and verify signal variance restoration.
