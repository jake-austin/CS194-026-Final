
Jake Austin
3034720807
Jake-austin@berkeley.edu
https://hambone4701.github.io/CS194-026-Final/

The notebook has everything. It is super easy to follow. The first project is in the notebook final_pt1, and the same for the second project.

The first project:
This has a basic toy problem where I simply create a row in matrices A, b for each equation introduced by the equations listed that we need to optimize for.
In order to do all this, I had to use scipy sparce matrices, as these matrices are mostly zeros but are enormous. To add entries to the matrix, I used the lil_matrix type, converting to a csr_matrix which has better runtimes for algebra. 
In the second part, we have more complex equations. This is all in the gradient_blend() function. Basically, the sums in the equation iterate over all pixels in the mask, iterate again over all 4 neighbors and we create an equation based on whether the 4 neighbor is in the mask as well or not. I basically iterated over all pixels in the mask and just did an if statement for each of the 4 neighbors to see if it is in the mask or not, and just add the correct equation for each case. Then we just do least squares. I also created an advanced gradient blend where I added new equations as well which were that pixels we are solving for need to look like the pixels in the source image whose gradient we are using. I would weight both sides by a small constant too to make sure that I wasn't placing too much weight on these equations, with the hope that our solved for pixels would sort of start to fade to the pixel values in the original image near the border, avoiding some of the color issues that I faced when I used images with different backgrounds.

The second project:
This has one main optimize function that takes in all the tensors that we are going to optimize based on. We create hooks in the VGG model that cache all the activations at all the layers we are interested in into lists that we read from and clear each time we run an iteration of gradient descent. We take those activations and pass them into our style and content loss functions which were described in the paper and are named content_loss and style_loss in the notebook. We simply create an Adam optimizer over the parameters in our starting image tensor and then just keep going forward, passing the activations to our loss function, back propagating on the loss, and updating the image.

