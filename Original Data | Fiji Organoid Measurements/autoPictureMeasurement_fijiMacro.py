from ij import IJ
import time 
import datetime

#User Input
seriesNumber = 			"25"
organoidLabel = 		"A"
finalOrganoidLabel = 	"XX"

directory = "/Users/wesleyjohanson/Documents/College Classes_UC Santa Barbara/Pitenis Lab/Image Analysis/Project 2 | Radial Expansion Velocity/"
CSVdirectory = directory + "organoid_measurements/"
TIFdirectory = directory + "tif_files/"

#Program Parameters
startupTime = 10
frameSleep = 3
zBreak = 10.
zBreakTime = 4
firstVideoDelay = 15

# Auto-Generated Filenames 
CSVfilename = "series" + seriesNumber + "_organoid_" + organoidLabel + ".csv"
TIFfilename = "series" + seriesNumber + "_masked_Ato" + organoidLabel + ".tif"
origTIFfilename = "series" + seriesNumber + ".tif"
finalTIFfilename =  "series" + seriesNumber + "_masked.tif"
alarm = "||||||||||||||||||||||||||||||||||||||||||||||||"

print("series " + seriesNumber + " organoid " + organoidLabel)

#Check if file already exists (no rewrite)
imp = IJ.openImage(TIFdirectory + TIFfilename);
if imp is not None: 
	print("ERROR")
	quit()


#Setup
#if organoidLabel == "A":
#	imp = IJ.openImage(TIFdirectory + origTIFfilename);
#	imp.show();
#	time.sleep(firstVideoDelay);
#else: #check if the letter is one more than that last video that was made
#	imp = IJ.getImage();
imp = IJ.getImage();
#Open in 8bit so it's faster 
#IJ.run("8-bit", "");
#IJ.run(imp, "8-bit", "");
	
Nframes = imp.getNFrames();
originalTitle = imp.getTitle();
imp.setPosition(imp.getC(), imp.getZ(), 1);
IJ.setTool("line");
#CHANGE THIS FOR DIFFERENT EXPERIEMENTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
IJ.run(imp, "Set Scale...", "distance=130.8914 known=166.98 unit=micron global");
IJ.run("Clear Results", "");


for i in range(startupTime, 0, -1):
#	print("starting in " + str(i))
	imp.setTitle("Recording first line measurement in ... " + str(i) + " seconds")
	time.sleep(1)


for i in range(1, Nframes + 1):
	#Record the distance of the line, for frame i
	imp.setPosition(imp.getC(), imp.getZ(), i);
	IJ.run(imp, "Measure", "");
	
	#Show the next frame (don't iterate)
	if i == Nframes: break
	imp.setPosition(imp.getC(),imp.getZ(),i + 1);
	
	#Additional Time Break to refocus the Z
	if i % zBreak == 0: 
		for t in range(zBreakTime, 0, -1):
			imp.setTitle(alarm + " REFOCUS Z | " + str(t) + " sec " + alarm)
			time.sleep(1)
	
	#Time break to record the frame
	for j in range(frameSleep, 0, -1):
		imp.setTitle(str(j) + " sec")
		time.sleep(1)
		


imp.setTitle(originalTitle)
	
#Save the Results measurements as a CSV file
IJ.saveAs("Results", CSVdirectory + CSVfilename);

#Save the tif file mask on it
IJ.saveAs(imp, "Tiff", TIFdirectory + TIFfilename);
if finalOrganoidLabel == organoidLabel:
	IJ.saveAs(imp, "Tiff", TIFdirectory + finalTIFfilename);

#Open the CSV File with MacOS
import subprocess
FileName = CSVdirectory + CSVfilename
subprocess.call(['open', FileName])

#Clear the Measurements for the next Experiment, reset the view
IJ.run("Clear Results", "");
IJ.run("View 100%", "");
#imp.setRoi(397,290,397,290);

#Close the windows 
#IJ.run("Close", "");
#imp.close();

print("PROGRAM COMPLETE")
