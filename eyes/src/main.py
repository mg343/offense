# eyes brain dump.

# can only be a sim for now, no livestream
# if full though, the functionality would involve one continous eval pipeline
# keeping many images loaded would be highly inefficient

# so, pipeline must be entirely contained within some decently timed loop - consider how fast the drone is moving.
# assuming the drone is up (100 feet seems reasonable)), and moving rather fast (20-40m/s), 
# should probably be near 30fps.

# within the 0.03 that allots per cycle, we should be able to:
# capture a frame from the camera
# pre-process it to make comparison accurate
# perform inference (compare input to sattelite imagery over target coords - have this loaded up for ease of comparison)
#     inference must be robust, big stakes, needs to be an effective strategy for inference
# make a decision via inference results
# drop frame
# restart loop

# we should keep the model as close to real world.
# thus, the input are limited to:
# 1. video stream (/files/stream.mp4 represents drone camera stream - this is suitable; most camera processing is straight math, which is near instantaneous calculations, so we can use this comfortably knowing that the amount of processing that would be associated with raw data would add negligible time),
# 2. coordinates of drop zone
#     2a. a straight image of the drop zone (technically easier, but tackiling the technically complex problem first)
#     2-note. some might argue that this assumes that drones will fly at the same altitude, which is unreasonable. however, pre-programmed/autonomous drones typically have a specified altitude at which they approach the target/reach target at.
#     for 2a., this need not be specified, dummy information can be given to the variable can be used. for 2.0, we will use a pre-programmed altitude variable, which is provided in addition to the coords to information to idneitfy the proper scale sattelite image to be retrieved, and apply saling if the altitude insists a "not clean" zoom.
#     all this to say - altitude changes are not very pertinent here, but we will build in the functionality for purposes of technical demonstration.

# libraries to be used (minimal, but for PoC don't have to go crazy. consider also, all hosted on machine, so data transmitting is not an issue)
# opencv for image processing
# openstreetmap, or any other open mapping software with a usable api for retrieving sattelite imagery
# opencv (or nothing, if reasonable) for inference - again, inference must be explainable and accurate. ai models might not be the correct solution here. consider more established mathematical approaches to comparing image similarity

# each major step should be broken out into its own function. inference will be the most heavy/important function, taking in a processed image from a stream, a sattelie image of the drop zone (which is procured via, as stated, its own function, ideally at initiatalization, and outputting a tuple of a GO/NO_GO with a confidence metric. threshold - 90%.)
# as a note - a go result does not neccessarily mean drop. it means eyes has triggered - the drone is over the target. assuming a high speed, this would mean a payload miss. the go signal should not be interpreted as a release payload sign. might have to reword later.
# the program is executed via a main/overall function that just calls all intermediary functions, and will be run in the typical if __name__ == __main__ setup.
#

print("Hello Drop Zone")