module TensorStream
  class Session

    def initialize(evaluator = TensorStream::RubyEvaluator)
      @evaluator_class = evaluator
    end

    def self.default_session
      @session ||= Session.new
    end

    def last_session_context
      @last_session_context
    end
  
    def run(*args)
      options = if args.last.kind_of?(Hash)
        args.pop
      else
        {}
      end
      context = {}

      #scan for placeholders and assign value
      options[:feed_dict].keys.each do |k|
        if k.kind_of?(Placeholder)
          context[k.name.to_sym] = options[:feed_dict][k].kind_of?(Tensor) ? options[:feed_dict][k] : TensorStream.constant(options[:feed_dict][k])
        end
      end if options[:feed_dict]
      evaluator = @evaluator_class.new(self, context.merge(retain: options[:retain]))
      
      execution_context = {}
      result = args.collect { |e| evaluator.eval(evaluator.eval(e, execution_context), execution_context) }
      @last_session_context = context
      result.size == 1 ? result.first : result
    end
  end
end