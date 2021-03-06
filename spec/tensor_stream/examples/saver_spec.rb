require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe "TensorStream::Train::Saver" do
  let(:tf) { TensorStream }
  it "saves models using the saver" do
    v1 = tf.get_variable("v1", shape: [3], initializer: tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape: [5], initializer: tf.zeros_initializer)

    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf::Train::Saver.new

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    tf.Session do |sess|
      sess.run(init_op)
      # Do some work with the model.
      inc_v1.run()
      dec_v2.run()
      # Save the variables to disk.
      save_path = saver.save(sess, "/tmp/model.ckpt")
      print("Model saved in path: %s" % save_path)
    end
  end

  it "restores variables using the saver" do
    tf.reset_default_graph()

    # Create some variables.
    v1 = tf.get_variable("v1", shape: [3])
    v2 = tf.get_variable("v2", shape: [5])

    # Add ops to save and restore all the variables.
    saver = tf::Train::Saver.new

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    tf.Session do |sess|
      # Restore variables from disk.
      saver.restore(sess, "/tmp/model.ckpt")
      print("Model restored.")
      # Check the values of the variables
      print("v1 : %s" % v1.eval())
      print("v2 : %s" % v2.eval())

      expect(v1.eval).to eq([1.0, 1.0, 1.0])
      expect(v2.eval).to eq([-1.0, -1.0, -1.0, -1.0, -1.0])
    end
  end
end