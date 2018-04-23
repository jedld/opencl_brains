require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream::Operation do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
    srand(1234)
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
          expect(TensorStream.random_uniform(shape: shape).eval).to eq(expected)
        end

        specify "with ranges" do
          expect(TensorStream.random_uniform(shape: shape, minval: 0, maxval: 2).eval).to eq(range_expected)
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
          expect(TensorStream.random_normal(shape: shape).eval).to eq(expected)
        end
      end
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
  [:sin, 0.09983341664682815, [[0.8912073600614354, -0.3820714171840091], [0.8632093666488737, 0.1411200080598672]],  0.9950041652780258],
  [:cos, 0.9950041652780258, [[0.4535961214255773, -0.9241328000731296], [-0.5048461045998576, -0.9899924966004454]], -0.09983341664682815],
  [:tan, 0.10033467208545055,  [[1.9647596572486523, 0.41343778421648336], [-1.7098465429045073, -0.1425465430742778]], 1.0100670464224948],
  [:tanh, 0.09966799462495582,  [[0.8004990217606297, 0.9999999999999792], [0.9704519366134539, 0.9950547536867305]], 0.9900662908474398],    
].each do |func, scalar, matrix, gradient|
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
      expect(sess.run(grad)).to eq(gradient)
    end
  end
end
 
  context ".derivative" do
    it "Creates a derivative graph for a computation" do
      x = TensorStream.placeholder(TensorStream::Types.float32)
      p = TensorStream.pow(x, 3) 

      derivative_function = TensorStream::Operation.derivative(p, x)
      expect(p.eval(feed_dict: { x => 2})).to eq(8)
      expect(derivative_function.eval(feed_dict: { x => 2})).to eq(12)
  
      # f(x) = (sin x) ^ 3
      # dx = 3(sin x)^2 * cos x
      y = TensorStream.sin(x) ** 3
      derivative_function_y = TensorStream::Operation.derivative(y, x)
      expect(derivative_function_y.eval(feed_dict: { x => 1 })).to eq(1.147721101851439)
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
  end

  context "combination of functions" do
    it "add two operation together" do
      y = TensorStream.sin(1) + TensorStream.sin(2)
      expect(y.eval).to eq(1.7507684116335782)
    end
  end
end