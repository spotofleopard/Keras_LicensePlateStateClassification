<img src='images/000247.jpg'><img src='images/003884.jpg'><img src='images/006116.jpg'><img src='images/007272.jpg'><img src='images/008467.jpg'><img src='images/010399.jpg'><br/>
<img src='images/012363.jpg'><img src='images/012720.jpg'><img src='images/015098.jpg'><img src='images/015532.jpg'><img src='images/016393.jpg'><img src='images/019203.jpg'>
<br/>
<p>Classification is one of the most popular tasks in deep learning. Here's a real world example of using this technology  to determine a US vehicle license plate's jurisdiction.</p>
<p>Here we used a ResNet50V2 as body, one single FC layer before softmax output. So how well does this simple model perform? It probably performs better than you expected. In my test, it at least matches and in many cases outperforms a well trained human operator(aka me). The very few cases it gets wrong result is when the plate is vanity plate with a totally different look absent in the training set.</p>

<img src='images/val_accuracy.png'>
