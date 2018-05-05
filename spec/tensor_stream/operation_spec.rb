require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Operation do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
    srand(1234)
  end

  let(:tf) { TensorStream } # allow calls to look like tensorflow

  context ".concat" do
    it "Concatenates tensors along one dimension." do
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]
      expect(tf.concat([t1, t2], 0).eval).to eq([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      expect(tf.concat([t1, t2], 1).eval).to eq([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]])
    end

    it "negative axis" do
      t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
      t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
      expect(tf.concat([t1, t2], -1).eval).to eq(
      [[[ 1,  2,  7,  4],
        [ 2,  3,  8,  4]], 
       [[ 4,  4,  2, 10],
        [ 5,  3, 15, 11]]])
    end
  end

  context ".reshape" do
    it "Reshapes a tensor." do
      t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(tf.reshape(t, [3, 3]).eval).to eq(
        [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

      t = [[[1, 1], [2, 2]],
           [[3, 3], [4, 4]]]

      expect(tf.reshape(t, [2, 4]).eval).to eq([[1, 1, 2, 2],
        [3, 3, 4, 4]])
    end

    it "reshape to scalar" do
      t = [7]
      expect(tf.reshape(t, []).eval).to eq(7)
    end

    it "flattens a tensor" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
          [[3, 3, 3],
          [4, 4, 4]],
          [[5, 5, 5],
          [6, 6, 6]]]
      expect(tf.shape(t).eval).to eq([3, 2, 3])
      expect(tf.reshape(t, [-1]).eval).to eq([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
      expect(tf.reshape(t, [2, -1]).eval).to eq([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]])
    end

    it "should fail if dimensions do not match" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
          [[3, 3, 3],
          [4, 4, 4]],
          [[5, 5, 5],
          [6, 6, 6]]]
      expect {
        tf.reshape(t,[3,2,2]).eval
      }.to raise_exception

    end

    it "inference" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
            [[3, 3, 3],
            [4, 4, 4]],
            [[5, 5, 5],
            [6, 6, 6]]]

      expect(tf.reshape(t, [-1, 9]).eval).to eq([[1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]])
      
      expect(tf.reshape(t, [ 2, -1, 3]).eval).to eq(
        [[[1, 1, 1],
          [2, 2, 2],
          [3, 3, 3]],
          [[4, 4, 4],
          [5, 5, 5],
          [6, 6, 6]]])
    end
  end

  context ".equal" do
    it "returns the truth value of two tensors" do
      a = TensorStream.constant(1.0)
      b = TensorStream.constant(1.0)
      c = TensorStream.constant(2.1)
      d = TensorStream.constant([[1.0]])
      e = TensorStream.constant([[1.0]])
      f = TensorStream.constant([[2.0]])
      expect(TensorStream.equal(a, b).eval).to eq(true)
      expect(TensorStream.equal(a, c).eval).to eq(false)
      expect(TensorStream.equal(d, e).eval).to eq(true)
      expect(TensorStream.equal(e, f).eval).to eq(false)

      expect((a == b).eval).to eq(true)
      expect((a == c).eval).to eq(false)
    end
  end

  # Outputs random values from a uniform distribution.
  # The generated values follow a uniform distribution in the range [minval, maxval). The lower bound minval is included in the range, while the upper bound maxval is excluded.
  # For floats, the default range is [0, 1). For ints, at least maxval must be specified explicitly.
  # In the integer case, the random integers are slightly biased unless maxval - minval is an exact power of two. The bias is small for values of maxval - minval significantly smaller than the range of the output (either 2**32 or 2**64).
  context ".random_uniform" do
    [
      [[],     0.1915194503788923,       0.3830389007577846         ],
      [[1],   [0.1915194503788923],      [0.3830389007577846]         ],
      [[2,3], [[0.1915194503788923, 0.6221087710398319, 0.4377277390071145], [0.7853585837137692, 0.7799758081188035, 0.2725926052826416]],  [[0.3830389007577846, 1.2442175420796637, 0.875455478014229], [1.5707171674275384, 1.559951616237607, 0.5451852105652832]] ]
    ].each do |shape, expected, range_expected|
      describe "shape #{shape}" do
        it "generates random uniform values" do
          expect(TensorStream.random_uniform(shape).eval).to eq(expected)
        end

        specify "with ranges" do
          expect(TensorStream.random_uniform(shape, minval: 0, maxval: 2).eval).to eq(range_expected)
        end
      end
    end
  end

  context ".random_normal" do
    [
      [[], 0.5011628459350929],
      [[1],   [0.5011628459350929] ],
      [[2,3], [[0.5011628459350929, 1.301972948852967, -1.621722019401658], [0.6690221526288901, 0.14937983113945622, -0.783723693080629]] ],
    ].each do |shape, expected|
      describe "shape #{shape}" do
        it "generates random normal values" do
          expect(TensorStream.random_normal(shape).eval).to eq(expected)
        end
      end
    end
  end

  context ".zeros" do
    it "generates a zero tensor" do
      a = tf.zeros([2,2])
      expect(a.eval).to eq([[0.0, 0.0], [0.0, 0.0]])
    end
  end

  context ".ones" do
    it "generates a ones tensor" do
      ones = tf.ones([2,2])
      expect(ones.eval).to eq([[1.0, 1.0], [1.0, 1.0]])
    end
  end

  context ".reduce_sum" do
    it "computes the sum of elements across dimensions of a tensor." do
      x = TensorStream.constant([[1, 1, 1], [1, 1, 1]])
      expect(TensorStream.reduce_sum(x).eval).to eq(6)
      expect(TensorStream.reduce_sum(x, 0).eval).to eq([2, 2, 2])
      expect(TensorStream.reduce_sum(x, 1).eval).to eq([3, 3])
      expect(TensorStream.reduce_sum(x, 1, keepdims: true).eval).to eq([[3], [3]])
      expect(tf.reduce_sum(x, [0, 1]).eval).to eq(6)
    end

    specify "computes the gradients properly" do
      a = tf.constant([[1,2,3],[4,5,6]])
      op = tf.reduce_sum(a)
      expect(tf.gradients(op,[a]).eval).to eq([[[1, 1, 1], [1, 1, 1]]])
    end
  end

  context ".pow" do
    it "Computes the power of tensor x to tensor y" do
      x = tf.constant([[2, 2], [3, 3]])
      y = tf.constant([[8, 16], [2, 3]])
      p = tf.pow(x, y)  # [[256, 65536], [9, 27]]
      sess = tf.Session
      expect(sess.run(p)).to eq([[256, 65536], [9, 27]])

      p = tf.pow(x, 2)
      expect(sess.run(p)).to eq([[4, 4], [9, 9]])
    end
  end

  context ".print" do
    it "behaves like identity but prints a message to stdout" do
      x = tf.constant([[2.0, 2.0], [3.0, 3.0]])
      y = tf.print(x, x, message: "this is a prefix")
      z = tf.sin(y)
      expect(tr(z.eval)).to eq([[0.9093, 0.9093], [0.1411, 0.1411]])
    end
  end

  context ".slice" do
    it "slices a tensor" do
      t = tf.constant([[[1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4]],
        [[5, 5, 5], [6, 6, 6]]])
      expect(tf.slice(t, [1, 0, 0], [1, 1, 3]).eval).to eq([[[3, 3, 3]]])
      expect(tf.slice(t, [1, 0, 0], [1, 2, 3]).eval).to eq([[[3, 3, 3], [4, 4, 4]]])
      expect(tf.slice(t, [1, 0, 0], [2, 1, 3]).eval).to eq([[[3, 3, 3]], [[5, 5, 5]]])
    end

    it "1D tensor slicing" do
      t  = tf.constant([1,2,3,4,5,6,7])
      expect(tf.slice(t, [2], [1]).eval).to eq([3])
    end
  end

  context ".rank" do
    it "returns the rank of a tensor" do
      t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
      t2 = tf.constant(1)
      t3 = tf.constant([1,2])
      rank1 = tf.rank(t1)
      rank2 = tf.rank(t2)
      rank3 = tf.rank(t3)
      expect(rank1.eval).to eq(3)
      expect(rank2.eval).to eq(0)
      expect(rank3.eval).to eq(1)
    end
  end

  context ".negate" do
    it "computes the negative of a tensor" do
      x = tf.constant(0.1)
      y = tf.constant([[1.1, 16.1], [2.1, 3.0]])
      z = -tf.constant(4.1)
      x_negate = tf.negate(x)
      y_negate = tf.negate(y)
      sess = tf.Session
      expect(sess.run(x_negate)).to eq(-0.1)
      expect(sess.run(y_negate)).to eq([[-1.1, -16.1], [-2.1, -3.0]])
      expect(sess.run(z)).to eq(-4.1)
    end
  end

  # tests for single parameter algebra functions
[
  [:sin, 0.0998,   [[0.8912, -0.3821], [0.8632, 0.1411]],  0.995, [[0.4536, -0.9241], [-0.5048, -0.99]]            ],
  [:cos, 0.995,    [[0.4536, -0.9241], [-0.5048, -0.99]], -0.0998, [[-0.8912, 0.3821], [-0.8632, -0.1411]]          ],
  [:tan, 0.1003,   [[1.9648, 0.4134], [-1.7098, -0.1425]], 1.0101,  [[4.8603, 1.1709], [3.9236, 1.0203]]          ],
  [:tanh, 0.0997,  [[0.8005, 1.0], [0.9705, 0.9951]],      0.9901, [[0.3592, 0.0], [0.0582, 0.0099]]                  ],
  [:log, -2.3026,  [[0.0953, 2.7788], [0.7419, 1.0986]],   10.0, [[0.9091, 0.0621], [0.4762, 0.3333]]              ],
  [:exp, 1.1052,   [[3.0042, 9820670.9221], [8.1662, 20.0855]], 1.1052, [[3.0042, 9820670.9221], [8.1662, 20.0855]]],
  [:square, 0.01,  [[1.21, 259.21], [4.41, 9.0]],          0.2, [[2.2, 32.2], [4.2, 6.0]]                                  ],
  [:negate, -0.1,  [[-1.1, -16.1], [-2.1, -3.0]],         -1.0, [[-1.0, -1.0], [-1.0, -1.0]]                              ],
  [:identity, 0.1, [[1.1, 16.1], [2.1, 3.0]],             1.0, [[1, 1], [1, 1]]                                              ],
  [:abs, 0.1,      [[1.1, 16.1], [2.1, 3.0]],             1.0, [[1, 1], [1, 1]]                                              ],
].each do |func, scalar, matrix, gradient, gradient2|
  context ".#{func}" do
    let(:x) { tf.constant(0.1) }
    let(:y) {  tf.constant([[1.1, 16.1], [2.1, 3.0]]) }
    let(:sess) { tf.Session }
    let(:f_x) { tf.send(func,x) }
    let(:f_y) { tf.send(func,y) }

    specify "scalar #{func} value" do
      expect(tr(sess.run(f_x))).to eq(scalar)
    end

    specify "matrix #{func} values" do
      expect(tr(sess.run(f_y))).to eq(matrix)
    end

    specify "gradient #{func} values" do
      grad = tf.gradients(f_x, [x])[0]
      grad_2 = tf.gradients(f_y, [y])[0]
      expect(tr(sess.run(grad))).to eq(gradient)
      expect(tr(sess.run(grad_2))).to eq(gradient2)
    end
  end
end

  context ".abs" do
    it "Computes the absolute value of a tensor" do
      tf = TensorStream

      a = [[1,2],[-1, 2], [3,-3]]
      b = -1.123

      expect(tf.abs(a).eval).to eq([[1, 2], [1, 2], [3, 3]])
      expect(tf.abs(b).eval).to eq(1.123)
    end

    specify "should compute for the gradient" do
      a = tf.constant([[1,2],[-1, 2], [3,-3]])
      expect(tf.gradients(tf.abs(a),[a]).eval).to eq([[[ 1,  1],
        [-1,  1],
        [ 1, -1]]])
    end
  end

  context ".sign" do
    it "Returns an element-wise indication of the sign of a number." do
      tf = TensorStream

      a = tf.constant([[1,2],[-1, 2], [3,-3]])
      b = -1.123

      expect(tf.sign(a).eval).to eq([[1, 1], [-1, 1], [1, -1]])
      expect(tf.sign(b).eval).to eq(-1.0)
    end
  end

  context ".matmul" do
    it "performs matrix multiplication" do
      a = tf.constant([1, 2, 3, 4, 5, 6], shape: [2, 3])
      b = tf.constant([7, 8, 9, 10, 11, 12], shape: [3, 2])
      c = tf.matmul(a, b)
      expect(c.eval).to eq([[ 58,  64],
                            [139, 154]])
      d = tf.matmul(a, b, transpose_a: true, transpose_b: true)
      expect(d.eval).to eq([[39, 49, 59], [54, 68, 82], [69, 87, 105]])
    end

    specify "gradients" do
      a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
      
      y = tf.matmul(a, tf.sin(b))
      g = tf.gradients(y, [a,b])

      expect(g.eval).to eq([
        [[ 2.0584633, -2.0805843, -2.0805843],
         [ 2.0584633, -2.0805843, -2.0805843]], 
        [[ 3.7695112 , -0.7275002 , -4.555651],
         [-5.873501  ,  0.03097989,  5.9069777 ],
         [-7.5516434 ,  0.03983128,  7.5946856 ]]])
    end
  end

  context ".transpose" do
    it "transposes matrices" do
      tf.program do |tf|
        x = tf.constant([[1, 2, 3], [4, 5, 6]])
        t = tf.transpose(x)
        sess = tf.Session
        expect(sess.run(t)).to eq([[1, 4], [2, 5], [3, 6]])
      end
    end
  end

  context ".derivative" do
    it "Creates a derivative graph for a computation" do
      x = tf.placeholder(TensorStream::Types.float32)
      p = tf.pow(x, 3) 

      derivative_function = TensorStream::MathGradients.derivative(p, x)
      expect(p.eval(feed_dict: { x => 2})).to eq(8)
      expect(derivative_function.eval(feed_dict: { x => 2})).to eq(12)
  
      # f(x) = (sin x) ^ 3
      # dx = 3(sin x)^2 * cos x
      y = tf.sin(x) ** 3
      derivative_function_y = TensorStream::MathGradients.derivative(y, x)
      expect(derivative_function_y.eval(feed_dict: { x => 1 })).to eq(1.147721101851439)
    end
  end

  context ".shape" do
    it "returns a 1D tensor representing shape of target tensor" do
      tf.program do |tf|
        t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
        shape = tf.shape(t)
        expect(shape.eval).to eq([2, 2, 3])

        u = tf.constant(1)
        shape = tf.shape(u)
        expect(shape.eval).to eq([])

        v = tf.constant([[1,2,3],[4,5,6]])
        shape = tf.shape(v)
        expect(shape.eval).to eq([2 ,3])
      end
    end
  end

  context ".eye" do
    it "creates an identity matrix" do
      tf.program do |tf|
        e = tf.eye(2)
        expect(e.eval).to eq([[1.0, 0.0],[0.0, 1.0]])

        e = tf.eye(3)
        expect(e.eval).to eq([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        e = tf.eye(3, num_columns: 2)
        expect(e.eval).to eq([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      end
    end
  end

  context ".gradients" do
    it "Constructs symbolic derivatives of sum of ys w.r.t. x in xs." do
      a = tf.constant(0.0)
      b = a * 2
      g = tf.gradients(a + b, [a, b], stop_gradients: [a, b])
      h = tf.gradients(a + b, [a, b])

      expect(g.eval).to eq([1.0, 1.0])
      expect(h.eval).to eq([3.0, 1.0])
    end

    it "using stop gradients" do
      a = tf.stop_gradient(tf.constant(0.0))
      b = tf.stop_gradient(a * 2)
      h = tf.gradients(a + b, [a, b])
      expect((a+b).eval).to eq(0)
      expect((a+b).to_math).to eq("(0.0 + (0.0 * 2.0))")
      expect(h.eval).to eq([1.0, 1.0])
    end

    it "computes gradient of sin" do
      var = tf.constant(1.0) # Must be a tf.float32 or tf.float64 variable.
      loss = tf.sin(var) # some_function_of() returns a `Tensor`.
      var_grad = tf.gradients(loss, [var])[0]

      expect(var_grad.eval).to eq(0.5403023058681398)
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

    it "should handle placeholders" do
      x = tf.placeholder("float", shape: [nil, 4])
      y = tf.placeholder("float", shape: [nil, 2])
      cz = tf.matmul(x, y)
      z_grad = tf.gradients(cz, [x, y])
      expect(tr(z_grad.eval(feed_dict: {
        x => [[1.0, 2.0 , 2.1, 0.8], [3.0, 4.0, 3.1, 0.9]],
        y => [[4.0, 5.0], [1.1, 3.2], [5.0, 3.1], [1.0, 1.0]]}))).to eq([[[9.0, 4.3, 8.1, 2.0], [9.0, 4.3, 8.1, 2.0]], [[4.0, 4.0], [6.0, 6.0], [5.2, 5.2], [1.7, 1.7]]])
    end
  end

  context ".cond" do
    it "returns a specific tensor function depending on the value of the predicate"  do
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = tf.multiply(x, y)

      result = tf.cond(x < y, tf.add(x, z), tf.square(y))
      result2 = tf.cond(x > y, -> { tf.add(x, z) }, -> { tf.square(y) })
      expect(result.eval).to eq(8.0)
      expect(result2.eval).to eq(9.0)
    end
  end

  context ".less" do
    it "returns true if a < b" do
      a = tf.constant(2.0)
      b = tf.constant(3.0)
      expect(tf.less(a, b).eval).to eq(true)
      expect(tf.less(b, a).eval).to eq(false)
    end
  end

  context ".greater" do
    it "returns true if a > b" do
      a = tf.constant(2.0)
      b = tf.constant(3.0)
      expect(tf.greater(a, b).eval).to eq(false)
      expect(tf.greater(b, a).eval).to eq(true)
    end
  end

  context ".sub" do
    let(:a) { tf.constant([1.0, 2.0, 3.0])}
    let(:b) { tf.constant([0.1, 0.2, 0.3])}
    let(:c) { tf.constant(0.1) }
    let(:m) { tf.constant([[1.0, 2.0, 3.0], [2.0, 3.0 ,4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]) }

    it "substracts two arrays" do
      expect((a - b).eval).to eq([0.9, 1.8, 2.7])
    end

    it "substracts an array and a constant" do
      expect((a - c).eval).to eq([0.9, 1.9, 2.9])
    end

    it "substracts a matrix and an array" do
      expect((m - a).eval).to eq([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [7.0, 7.0, 7.0]])
    end
  end

  context ".div" do
    let(:a) { tf.constant(2.5) }
    let(:b) { tf.constant(3.1) }

    it "divides to tensors" do
      op = a / b
      expect(tr(op.eval)).to eq(0.8065)
    end

    it "supports gradients" do
      grad = tf.gradients(a/b, [a,b])
      expect(tr(grad.eval)).to eq([0.3226, -0.2601])
    end
  end

  context "combination of functions" do
    it "add two operation together" do
      y = tf.sin(1.0) + tf.sin(2.0)
      expect(y.eval).to eq(1.7507684116335782)
    end
  end
end