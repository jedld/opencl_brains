module TensorStream
  class Session

    def initialize(evaluator = TensorStream::RubyEvaluator)
      @evaluator_class = evaluator
    end

    def self.default_session
      @session ||= Session.new
    end
    def run(*args)

      options = if args.last.kind_of?(Hash)
        args.pop
      else
        {}
      end
      context = {}
      evaluator = @evaluator_class.new(self, context)
      #scan for placeholders and assign value
      options[:feed_dict].keys.each do |k|
        if k.kind_of?(Placeholder)
          context[k.name.to_sym] = options[:feed_dict][k].kind_of?(Tensor) ? options[:feed_dict][k] : TensorStream.constant(options[:feed_dict][k])
        end
      end if options[:feed_dict]

      result = args.collect { |e| evaluator.eval(evaluator.eval(e)) }

      result.size == 1 ? result.first : result
    end
  end
end