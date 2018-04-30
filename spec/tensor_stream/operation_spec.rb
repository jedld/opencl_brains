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
      expect(TensorStream.reduce_sum(x, [0, 1]).eval).to eq(6)
    end

    specify "computes the gradients properly" do
      a = tf.constant([[1,2,3],[4,5,6]])
      op = tf.reduce_sum(a)
      expect(tf.gradients(op,[a]).eval).to eq([[[1, 1, 1], [1, 1, 1]]])
    end
  end

  context ".pow" do
    it "Computes the power of tensor x to tensor y" do
      x = TensorStream.constant([[2, 2], [3, 3]])
      y = TensorStream.constant([[8, 16], [2, 3]])
      p = TensorStream.pow(x, y)  # [[256, 65536], [9, 27]]
      sess = TensorStream.Session
      expect(sess.run(p)).to eq([[256, 65536], [9, 27]])

      p = TensorStream.pow(x, 2)
      expect(sess.run(p)).to eq([[4, 4], [9, 9]])
    end
  end

  context ".negate" do
    it "computes the negative of a tensor" do
      x = TensorStream.constant(0.1)
      y = TensorStream.constant([[1.1, 16.1], [2.1, 3.0]])
      z = -TensorStream.constant(4.1)
      x_negate = TensorStream.negate(x)
      y_negate = TensorStream.negate(y)
      sess = TensorStream.Session
      expect(sess.run(x_negate)).to eq(-0.1)
      expect(sess.run(y_negate)).to eq([[-1.1, -16.1], [-2.1, -3.0]])
      expect(sess.run(z)).to eq(-4.1)
    end
  end

[
  [:sin, 0.09983341664682815, [[0.8912073600614354, -0.3820714171840091], [0.8632093666488737, 0.1411200080598672]],  0.9950041652780258, [[0.4535961214255773, -0.9241328000731296], [-0.5048461045998576, -0.9899924966004454]]],
  [:cos, 0.9950041652780258, [[0.4535961214255773, -0.9241328000731296], [-0.5048461045998576, -0.9899924966004454]], -0.09983341664682815, [[-0.8912073600614354, 0.3820714171840091], [-0.8632093666488737, -0.1411200080598672]]],
  [:tan, 0.10033467208545055,  [[1.9647596572486523, 0.41343778421648336], [-1.7098465429045073, -0.1425465430742778]], 1.0100670464224948,  [[4.860280510751841, 1.1709308014178355], [3.923575200282495, 1.020319516942427]]],
  [:tanh, 0.09966799462495582,  [[0.8004990217606297, 0.9999999999999792], [0.9704519366134539, 0.9950547536867305]], 0.9900662908474398, [[0.35920131616027484, 4.1522341120980855e-14], [0.058223038723196896, 0.009866037165440211]]],
  [:log, -2.3025850929940455,  [[0.09531017980432493, 2.7788192719904172], [0.7419373447293773, 1.0986122886681098]], 10.0, [[0.9090909090909091, 0.06211180124223602], [0.47619047619047616, 0.3333333333333333]]],
  [:exp, 1.1051709180756477,  [[3.0041660239464334, 9820670.922071371], [8.166169912567652, 20.085536923187668]], 1.1051709180756477, [[3.0041660239464334, 9820670.922071371], [8.166169912567652, 20.085536923187668]]],
].each do |func, scalar, matrix, gradient, gradient2|
  context ".#{func}" do
    it "Computes for the #{func} of a tensor" do
      x = TensorStream.constant(0.1)
      y = TensorStream.constant([[1.1, 16.1], [2.1, 3.0]])
      x_sin = TensorStream.send(func,x)
      y_sin = TensorStream.send(func,y)

      sess = TensorStream.Session
      expect(sess.run(x_sin)).to eq(scalar)
      expect(sess.run(y_sin)).to eq(matrix)

      grad = TensorStream.gradients(x_sin, [x])[0]
      grad_2 = TensorStream.gradients(y_sin, [y])[0]
      expect(sess.run(grad)).to eq(gradient)
      expect(sess.run(grad_2)).to eq(gradient2)
    end
  end
end

  context ".abs" do
    let(:tf) { TensorStream }
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
    let(:tf) { TensorStream }
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
      tf = TensorStream

      a = tf.constant([1, 2, 3, 4, 5, 6], shape: [2, 3])
      b = tf.constant([7, 8, 9, 10, 11, 12], shape: [3, 2])
      c = tf.matmul(a, b)
      expect(c.eval).to eq([[ 58,  64],
                            [139, 154]])
      d = tf.matmul(a, b, transpose_a: true, transpose_b: true)
      expect(d.eval).to eq([[39, 49, 59], [54, 68, 82], [69, 87, 105]])
    end
  end

  context ".transpose" do
    it "transposes matrices" do
      TensorStream.program do |tf|
        x = tf.constant([[1, 2, 3], [4, 5, 6]])
        t = tf.transpose(x)
        sess = tf.Session
        expect(sess.run(t)).to eq([[1, 4], [2, 5], [3, 6]])
      end
    end
  end

  context ".derivative" do
    it "Creates a derivative graph for a computation" do
      x = TensorStream.placeholder(TensorStream::Types.float32)
      p = TensorStream.pow(x, 3) 

      derivative_function = TensorStream::MathGradients.derivative(p, x)
      expect(p.eval(feed_dict: { x => 2})).to eq(8)
      expect(derivative_function.eval(feed_dict: { x => 2})).to eq(12)
  
      # f(x) = (sin x) ^ 3
      # dx = 3(sin x)^2 * cos x
      y = TensorStream.sin(x) ** 3
      derivative_function_y = TensorStream::MathGradients.derivative(y, x)
      expect(derivative_function_y.eval(feed_dict: { x => 1 })).to eq(1.147721101851439)
    end
  end

  context ".shape" do
    it "returns a 1D tensor representing shape of target tensor" do
      TensorStream.program do |tf|
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
      TensorStream.program do |tf|
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
      a = TensorStream.constant(0.0)
      b = a * 2
      g = TensorStream.gradients(a + b, [a, b], stop_gradients: [a, b])
      h = TensorStream.gradients(a + b, [a, b])

      expect(g.eval).to eq([1.0, 1.0])
      expect(h.eval).to eq([3.0, 1.0])
    end

    it "using stop gradients" do
      a = TensorStream.stop_gradient(TensorStream.constant(0.0))
      b = TensorStream.stop_gradient(a * 2)
      h = TensorStream.gradients(a + b, [a, b])
      expect((a+b).eval).to eq(0)
      expect((a+b).to_math).to eq("(0.0 + (0.0 * 2))")
      expect(h.eval).to eq([1.0, 1.0])
    end

    it "computes gradient of sin" do
      var = TensorStream.constant(1.0)              # Must be a tf.float32 or tf.float64 variable.
      loss = TensorStream.sin(var)  # some_function_of() returns a `Tensor`.
      var_grad = TensorStream.gradients(loss, [var])[0]

      expect(var_grad.eval).to eq(0.5403023058681398)
    end


      it "computes for the derivative of a matrix multiplication operation" do
        tf = TensorStream
        y =   tf.constant([[1.0,2.0],[3.0,4.0]], dtype: :float32)
        x = tf.constant([[4.0,5.0],[5.0,6.0]], dtype: :float32)
       
        c = tf.matmul(x, y)
       
        expect(c.eval).to eq([[19, 28], [23, 34]])
        c_grad = tf.gradients(c, [x, y])
        expect(c_grad.eval).to eq([
          [[3.0, 7.0], [3.0, 7.0 ]],
          [[9.0, 9.0], [11.0, 11.0]]
        ])  
      end

      it "should properly handle the gradient of non cubic matrices" do
        y = tf.constant([[1.0,2.0],[3.0,4.0]], dtype: :float32)
        z = tf.constant([[4.0,5.0]], dtype: :float32)
        cz = tf.matmul(z, y)
        z_grad = tf.gradients(cz, [y])
        expect(z_grad.eval).to eq([
          [[4.0, 4.0], [5.0, 5.0]]
        ])
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
      y = TensorStream.sin(1) + TensorStream.sin(2)
      expect(y.eval).to eq(1.7507684116335782)
    end
  end
end