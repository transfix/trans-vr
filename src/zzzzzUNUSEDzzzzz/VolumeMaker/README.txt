.----------------.
|VolumeMaker-0.1a| - Joe R. (transfix@ices.utexas.edu) - 05/12/2008
'================'

Pre-release version... now included as part of VolRover.
Usage is pretty straightforward.  Things to note:

- When you change things in the UI, they will modify the volume directly,
  hence the lack of a save option.  With this in mind, ALWAYS work with
  a copy of your original.  It's not my fault if your data breaks :)

- Modifying a volume's dimension results in a resizing of the volume image
  data using trilinear interpolation.

- The bounding box is used to define the volume in object space...

- The variable list shows each variables timestep if you double click on their
  names.  Every variable has the same number of timesteps in a single
  volume file.

- Re-mapping voxel values is useful when you need all your values within a
  specific range, such as when you're changing the datatype from a floating
  point type to an integral type.  For example, when going from float->uchar,
  you should first map your data to (0.0,255.0) in order to minimize loss of
  precision.

If you have questions, just drop me an e-mail!

