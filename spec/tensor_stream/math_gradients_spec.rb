require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::MathGradients do
  let(:tf) { TensorStream }

  context "addition" do
    it "handles shape differences, rank 2 vs 1" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant([1, 1])
      sum = a + b
      g = tf.gradients(sum, [a, b])

      expect(g.eval).to eq([[[1, 1], [1, 1], [1, 1]], [3, 3]])
    end

    it "handles shape differences, rank 2 vs 0" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant(1)
      sum = a + b
      g = tf.gradients(sum, [a, b])

      expect(g.eval).to eq([[[1, 1], [1, 1], [1, 1]], 6])
    end
  end

  it "computes for the derivative of a matrix multiplication operation" do
    y = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype: :float32)
    x = tf.constant([[4.0, 5.0], [5.0, 6.0]], dtype: :float32)

    c = tf.matmul(x, y)

    expect(c.eval).to eq([[19, 28], [23, 34]])
    c_grad = tf.gradients(c, [x, y])
    expect(c_grad.eval).to eq([
      [[3.0, 7.0], [3.0, 7.0]],
      [[9.0, 9.0], [11.0, 11.0]]
    ])
  end

  it 'should properly handle the gradient of non cubic matrices' do
    y = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype: :float32)
    z = tf.constant([[4.0, 5.0]], dtype: :float32)
    cz = tf.matmul(z, y)
    z_grad = tf.gradients(cz, [y])
    expect(z_grad.eval).to eq([
      [[4.0, 4.0], [5.0, 5.0]]
    ])
  end

  it 'should handle matrix gradients with incompatible transpositions' do
    y = tf.constant([[1.0, 2.0 , 2.1, 0.8], [3.0, 4.0, 3.1, 0.9]], dtype: :float32)
    z = tf.constant([[4.0, 5.0], [1.1, 3.2], [5.0, 3.1], [1.0, 1.0]], dtype: :float32)
    cz = tf.matmul(y, z)
    expect(tr(cz.eval)).to eq([[17.5, 18.71], [32.8, 38.31]])
    z_grad = tf.gradients(cz, [y, z])
    expect(tr(z_grad.eval)).to eq(
      [[[9.0 , 4.3, 8.1, 2.0 ],
        [9.0 , 4.3, 8.1, 2.0 ]], 

        [
          [4.0 , 4.0 ],
          [6.0 , 6.0 ],
          [5.2, 5.2 ],
          [1.7, 1.7 ]]])
  end

  context "placeholders" do
    let(:test_inputs) {
      [
        [0.5937, 0.2343, 1.4332, 0.4395],
        [-1.0227, -0.6915, 1.2367, 0.3452],
        [-0.5675, 1.0374, 1.0429, 0.8839],
        [-0.1066, -0.0469, -1.6317, -1.4836],
        [0.7835, -3.0105, 1.713, -0.4536],
        [-0.3076, 1.3662, -0.6537, 0.0905],
        [-0.2459, 0.2243, -2.7048, 0.848],
      ]
    }

    it "should handle placeholders" do
      x = tf.placeholder("float", shape: [nil, 4])
      y = tf.placeholder("float", shape: [nil, 2])
      cz = tf.matmul(x, y)
      z_grad = tf.gradients(cz, [x, y])
      expect(tr(z_grad.eval(feed_dict: {
        x => [[1.0, 2.0 , 2.1, 0.8], [3.0, 4.0, 3.1, 0.9]],
        y => [[4.0, 5.0], [1.1, 3.2], [5.0, 3.1], [1.0, 1.0]]}))).to eq([[[9.0, 4.3, 8.1, 2.0], [9.0, 4.3, 8.1, 2.0]], [[4.0, 4.0], [6.0, 6.0], [5.2, 5.2], [1.7, 1.7]]])
    end

    it "neural net gradients" do
      num_inputs = 4
      num_neurons = 5
      inputs = tf.placeholder("float", shape: [nil, num_inputs])
      biases = tf.constant([0.5012, 1.302, -1.6217, 0.669, 0.1494])

      weights = tf.constant([
        [-0.9135, 1.0376, 0.8537, 0.4376, 1.3255],
        [-0.5921, -1.4081, 1.0614, -0.5283, 1.1832],
        [0.7285, -0.7844, 0.1793, -0.5275, -0.4426],
        [-1.4976, 0.4433, 2.2317, -2.0479, 0.7791]])
      
      matrix_mul = tf.matmul(inputs, weights)
      sess = tf.Session()
      output = sess.run(matrix_mul, feed_dict: { inputs => test_inputs })
      expect(tr(output)).to eq(
        [ [-0.2952, -0.6433, 1.9933, -1.52, 0.7723],
          [1.7276, -0.9045, -0.6149, -1.4415, -2.4522],
          [-0.6598, -2.4758, 2.7762, -3.1567, 0.7023],
          [1.1583, 0.5777, -3.7443, 3.8771, -0.6305],
          [2.994, 3.5073, -3.2316, 1.9586, -3.6351],
          [-1.1397, -1.69, 1.2722, -0.6969, 1.5686],
          [-3.1486, 1.9266, 1.4357, -0.5359, 1.7973]]
      )

      neural_net =  matrix_mul + biases

      output = sess.run(neural_net, feed_dict: { inputs => test_inputs })

      expect(tr(output)).to eq(
        [
          [0.206, 0.6587, 0.3716, -0.851, 0.9217],
          [2.2288, 0.3975, -2.2366, -0.7725, -2.3028],
          [-0.1586, -1.1738, 1.1545, -2.4877, 0.8517],
          [1.6595, 1.8797, -5.366, 4.5461, -0.4811],
          [3.4952, 4.8093, -4.8533, 2.6276, -3.4857],
          [-0.6385, -0.388, -0.3495, -0.0279, 1.718],
          [-2.6474, 3.2286, -0.186, 0.1331, 1.9467]
        ]
      )

      g = tf.gradients(neural_net, [weights, biases])
      weight_gradient, biases_gradient = sess.run(g, feed_dict: { inputs => test_inputs })
      expect(tr(weight_gradient)).to eq([
        [-0.8731, -0.8731, -0.8731, -0.8731, -0.8731],
        [-0.8867, -0.8867, -0.8867, -0.8867, -0.8867],
        [0.4356, 0.4356, 0.4356, 0.4356, 0.4356],
        [0.6699, 0.6699, 0.6699, 0.6699, 0.6699]]
      )
      expect(tr(biases_gradient)).to eq([7.0, 7.0, 7.0, 7.0, 7.0])
     end
  end
end