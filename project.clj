(defproject ml-test "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.9.1"]
                 [org.nd4j/nd4j-native-platform "0.9.1"]
                 [hswick/jutsu.matrix "0.0.14"]]
  :main ^:skip-aot ml-test.xor
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
