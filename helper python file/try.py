image1= Image(path1)
image2= Image(path2)
image3= Image(path3)
image4= Image(path4)

mixer=Mixer(image1,image2)

image = mixer.mix()
image.save(path)