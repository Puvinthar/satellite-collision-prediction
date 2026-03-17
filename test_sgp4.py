from sgp4.api import Satrec
import math

line1 = "1 22675U 93036A   26075.20837901  .00000089  00000+0  41827-4 0  9995"
line2 = "2 22675  74.0395 145.1072 0025342 209.4962 150.4761 14.33236317711129"
satrec = Satrec.twoline2rv(line1, line2)

print("e:", satrec.ecco)
print("i:", math.degrees(satrec.inclo))
print("raan:", math.degrees(satrec.nodeo))
print("argp:", math.degrees(satrec.argpo))
print("mA:", math.degrees(satrec.mo))
print("mm:", satrec.no_kozai * 1440 / (2*math.pi))
