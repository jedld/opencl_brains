module TensorStream
  class Session
    def run(operation)
      operation.ruby_eval
    end
  end
end