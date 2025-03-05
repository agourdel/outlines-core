//! Building an `Index` to efficiently map vocabulary tokens to state transitions.

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use regex_automata::util::alphabet::ByteClasses;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::prelude::*;
use crate::vocabulary::Vocabulary;
use crate::{Error, Result};

/// `Index` efficiently maps vocabulary tokens to state transitions.
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct Index {
    /// The ID of the initial state in the automaton, processing begins from this state.
    initial_state: StateId,
    /// A collection of states considered as terminal states.
    final_states: HashSet<StateId>,
    /// A mapping of state transitions, defined by tokens ids and their corresponding state changes.
    ///
    /// ### Example
    /// ```ignore
    /// transitions = {
    ///    1: {10: 2, 15: 3},
    ///    2: {20: 4, 25: 3},
    ///    3: {30: 4},
    ///    4: {40: 4},
    /// }
    ///  +--------------------------------------+
    ///  |               State 1                |
    ///  |            Initial State             |
    ///  +--------------------------------------+
    ///              |                     |
    ///              +                     |
    ///         Token ID 10                |
    ///  +-----------------------+         |
    ///  |        State 2        |         |
    ///  +-----------------------+         |
    ///       |             |              |
    ///       |             +              +
    ///       |        Token ID 25    Token ID 15
    ///       |        +------------------------+
    ///       |        |        State 3         |
    ///       |        +------------------------+
    ///       |                            |
    ///       +                            +
    ///  Token ID 20                  Token ID 30
    ///  +--------------------------------------+
    ///  |               State 4                |
    ///  |             Final state              |
    ///  +--------------------------------------+
    /// ```
    transitions: HashMap<StateId, HashMap<TokenId, StateId>>,
    /// The token ID reserved for the "end-of-sequence" token.
    eos_token_id: TokenId,
}
/// The `Index` structure is designed to efficiently map tokens from a given vocabulary
/// to state transitions within a finite-state automaton.
///
/// ## Usage:
/// The `Index` is typically constructed by combining a vocabulary and regular expressions.
/// Once built, it can be used to efficiently evaluate token sequences or to validate input data.
///
/// ## Example:
/// ```rust
/// use outlines_core::prelude::*;
///
/// # fn run() -> Result<(), outlines_core::Error> {
/// let regex = "0|[1-9][0-9]*";
/// let vocabulary = Vocabulary::from_pretrained("openai-community/gpt2", None)?;
/// let index = Index::new(regex, &vocabulary)?;
///
/// let initial_state = index.initial_state();
/// println!("Initial state is {}", initial_state);
/// println!("Is initial state a final state? {}", index.is_final_state(&initial_state));
///
/// let allowed_tokens = index.allowed_tokens(&initial_state).expect("Some allowed tokens");
/// println!("Allowed tokens at initial state are {:?}", allowed_tokens);
///
/// let token_id = allowed_tokens.first().expect("First token");
/// println!("Next state for the token_id {} is {:?}", token_id, index.next_state(&initial_state, token_id));
///
/// println!("Final states are {:?}", index.final_states());
/// println!("Index has exactly {} transitions", index.transitions().len());
/// # Ok(())
/// # }
///
/// ```
///
/// ## Performance:
/// - **Complexity**:
///   The `Index` can accommodate large vocabularies and complex regular expressions.
///   However, its size may grow significantly with the complexity of the input.
/// - **Construction Cost**:
///   Building the `Index` involves processing the vocabulary and regular expressions,
///   which may require a considerable amount of time and computational resources.
impl Index {
    /// Builds an `Index` from regular expression and vocabulary tokens.
    pub fn new(regex: &str, vocabulary: &Vocabulary) -> Result<Self> {
        let eos_token_id = vocabulary.eos_token_id();
        let dfa = DFA::new(regex).map_err(Box::new)?;
        let start_state = match dfa.universal_start_state(Anchored::Yes) {
            Some(s) => s,
            None => return Err(Error::DfaHasNoStartState),
        };

        let mut transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
        let mut final_states: HashSet<StateId> = HashSet::default();

        let mut seen: HashSet<AutomataStateId> = HashSet::from_iter([start_state]);
        let mut next_states: Vec<AutomataStateId> = vec![start_state];

        while let Some(current_state) = next_states.pop() {
            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state.as_u32());
            }

            'token_loop: for (token, ids) in vocabulary.tokens().iter() {
                if ids.contains(&eos_token_id) {
                    continue;
                }

                let mut next_state = current_state;
                for transition_byte in token {
                    next_state = dfa.next_state(next_state, *transition_byte);
                    if dfa.is_dead_state(next_state) || dfa.is_quit_state(next_state) {
                        continue 'token_loop;
                    }
                }

                let is_intermediate_state = !dfa.is_match_state(next_state);
                let is_full_match_state = dfa.is_match_state(dfa.next_eoi_state(next_state));
                if is_intermediate_state || is_full_match_state {
                    for token_id in ids {
                        transitions
                            .entry(current_state.as_u32())
                            .or_default()
                            .insert(*token_id, next_state.as_u32());
                    }
                }
                if !seen.contains(&next_state) {
                    seen.insert(next_state);
                    next_states.push(next_state);
                }
            }
        }

        // Populate `transitions` with mappings from `final_states` to `eos_token_id`
        for &final_state in &final_states {
            transitions
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
        }

        Ok(Self {
            initial_state: start_state.as_u32(),
            final_states,
            transitions,
            eos_token_id,
        })
    }

    /// Builds an optimized `Index` using two-level token pre-classification.
    pub fn new_optimized(regex: &str, vocabulary: &Vocabulary) -> Result<Self> {
        let eos_token_id = vocabulary.eos_token_id();
        let vocab_size = vocabulary.tokens().len();

        
        let dfa = DFA::new(regex).map_err(Box::new)?;
        let start_state = dfa.universal_start_state(Anchored::Yes)
            .ok_or(Error::DfaHasNoStartState)?;

       
        // Structures conformes à Index
        let mut transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
        let mut final_states: HashSet<StateId> = HashSet::default();
        let mut state_map: HashMap<AutomataStateId, StateId> = HashMap::default();

        let byte_classes = dfa.byte_classes();

        // Pré-classement des tokens sur deux niveaux : premier byte, puis deuxième byte
        let mut token_groups: HashMap<u32, HashMap<u32, Vec<(Vec<u8>, Vec<TokenId>)>>> = HashMap::default();
        token_groups.reserve(256);


        for (token, ids) in vocabulary.tokens().iter() {
            if !ids.contains(&eos_token_id) && !token.is_empty() {
                // Utiliser la classe d'équivalence du premier octet
                let first_class = byte_classes.get(token[0]) as u32;
                
                // Utiliser la classe d'équivalence du deuxième octet si disponible
                let second_class = if token.len() > 1 {
                    byte_classes.get(token[1]) as u32
                } else { 
                    0 
                };
                
                token_groups
                    .entry(first_class)
                    .or_default()
                    .entry(second_class)
                    .or_default()
                    .push((token.clone(), ids.clone()));
            }
        }

        for second_level in token_groups.values_mut() {
            for tokens in second_level.values_mut() {
                tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // Plus long au plus court
            }
        }


        let mut seen: HashSet<AutomataStateId> = HashSet::from_iter([start_state]);
        let mut next_states: Vec<AutomataStateId> = vec![start_state];
        let mut state_counter: StateId = 0;

        while let Some(current_state) = next_states.pop() {
            let current_state_id = *state_map.entry(current_state)
                .or_insert_with(|| {
                    state_counter += 1;
                    state_counter - 1
                });

            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state_id);
                continue; 
            }



            // Exploration par groupes de tokens (niveau 1 : premier byte)
            for (first_class, second_level) in &token_groups {
                let representative_byte = get_representative_byte(byte_classes, *first_class);
                let first_state = dfa.next_state(current_state, representative_byte);
                if dfa.is_dead_state(first_state) || dfa.is_quit_state(first_state) {
                    continue;
                }

                // Niveau 2 : deuxième byte
                for (second_class, tokens) in second_level {
                    let mut token_next_state = first_state;
                    let mut second_byte: u8 = 0;
                    if *second_class != 0 { // Si le token a un 2e byte
                        second_byte = get_representative_byte(byte_classes, *second_class);
                        token_next_state = dfa.next_state(token_next_state, second_byte);
                        if dfa.is_dead_state(token_next_state) || dfa.is_quit_state(token_next_state) {
                            continue;
                        }
                    }

                    // Tester les tokens du sous-groupe
                    for (token, ids) in tokens {
                        let mut final_next_state = token_next_state;
                        let mut valid = true;
                        for &byte in &token[if second_byte != 0 { 2 } else { 1 }..] { // À partir du 3e byte ou 2e si pas de 2e byte
                            final_next_state = dfa.next_state(final_next_state, byte);
                            if dfa.is_dead_state(final_next_state) || dfa.is_quit_state(final_next_state) {
                                valid = false;
                                break;
                            }
                        }

                        if !valid { continue;}

                        let is_intermediate_or_match = !dfa.is_match_state(final_next_state) ||
                            dfa.is_match_state(dfa.next_eoi_state(final_next_state));


                        if is_intermediate_or_match {
                            let next_state_id = *state_map.entry(final_next_state)
                                .or_insert_with(|| {
                                    state_counter += 1;
                                    state_counter - 1
                                });
                            for &token_id in ids {
                                transitions
                                    .entry(current_state_id)
                                    .or_default()
                                    .insert(token_id, next_state_id);
                            }
                            if !seen.contains(&final_next_state) {
                                seen.insert(final_next_state);
                                next_states.push(final_next_state);
                            }
                        }
                    }
                }
            }
        }

        // Ajout des transitions EOS pour les états finaux
        for &final_state in &final_states {
            transitions
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
        }

        Ok(Self {
            initial_state: state_map[&start_state],
            final_states,
            transitions,
            eos_token_id,
        })
}
    
    /// Returns the ID of the initial state in the automaton.
    pub fn initial_state(&self) -> StateId {
        self.initial_state
    }

    /// Returns set of final states.
    pub fn final_states(&self) -> &HashSet<StateId> {
        &self.final_states
    }

    /// Returns state transitions map of tokens ids and their corresponding transition states.
    pub fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>> {
        &self.transitions
    }

    /// Checks if state is in final states set or not.
    pub fn is_final_state(&self, state: &StateId) -> bool {
        self.final_states.contains(state)
    }

    /// Lists allowed tokens for a give state ID or `None` if it is not found in `Index`.
    pub fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>> {
        self.transitions
            .get(state)
            .map(|res| res.keys().cloned().collect())
    }

    /// Returns transition state for a given state and token id or `None` otherwise.
    pub fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        if token_id == &self.eos_token_id {
            return None;
        }
        Some(*self.transitions.get(state)?.get(token_id)?)
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index object with transitions:")?;
        for (state_id, token_ids) in self.transitions.iter() {
            writeln!(f, "{:?} -> {:#?}", state_id, token_ids)?;
        }
        Ok(())
    }
}

fn get_representative_byte(classes: &ByteClasses, class: u32) -> u8 {
    for byte in 0..=255u8 {
        if (classes.get(byte) as u32) == class {
            return byte;
        }
    }
    panic!("Impossible de trouver un octet représentatif pour la classe");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use std::io::Write;

    impl Vocabulary {
        fn token_by_id(&self, token_id: TokenId) -> Option<&Vec<u8>> {
            self.tokens().iter().find_map(|(token, ids)| {
                if ids.contains(&token_id) { Some(token) } else { None }
            })
        }
       
    }

    impl Index {
        fn generate_all_texts(&self, vocabulary: &Vocabulary) -> HashSet<String> {
            let mut results = HashSet::default();
            let mut current_text = Vec::new();
            self.generate_from_state(vocabulary, self.initial_state, &mut current_text, &mut results);
            results
        }
    
        fn generate_from_state(
            &self,
            vocabulary: &Vocabulary,
            state: StateId,
            current_text: &mut Vec<u8>,
            results: &mut HashSet<String>,
        ) {
            // Si l'état est final, ajouter le texte actuel comme résultat
            if self.final_states.contains(&state) {
                results.insert(String::from_utf8(current_text.clone()).unwrap_or_else(|_| {
                    String::from("Invalid UTF-8")
                }));
            }
    
            // Explorer toutes les transitions possibles depuis cet état
            if let Some(transitions_from_state) = self.transitions.get(&state) {
                for (&token_id, &next_state) in transitions_from_state {
                    if token_id == self.eos_token_id {
                        // EOS termine la génération, déjà couvert par final_states
                        continue;
                    }
                    if let Some(token_bytes ) = vocabulary.token_by_id(token_id) {
                        current_text.extend_from_slice(token_bytes);
                        self.generate_from_state(vocabulary, next_state, current_text, results);
                        current_text.truncate(current_text.len() - token_bytes.len());
                    }
                }
            }
        }

        fn count_total_transitions(&self) -> usize {
            self.transitions.values().map(|inner| inner.len()).sum()
        }

        fn count_total_states(&self) -> usize {
            self.transitions.values().count()
        }
    }

    #[test]
    fn index_from_regex() {
        let regex = "0|[1-9][0-9]*";
        let eos_token_id = 4;
        let mut vocabulary = Vocabulary::new(eos_token_id);
        for (token, token_id) in [("blah", 0), ("1a", 1), ("2", 2), ("0", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let initial_state = index.initial_state();
        assert_eq!(initial_state, 40);
        assert_eq!(index.final_states(), &HashSet::from_iter([24, 48, 56]));
        assert!(!index.is_final_state(&initial_state));

        let expected = HashMap::from_iter([
            (24, HashMap::from_iter([(3, 24), (4, 24), (2, 24)])),
            (48, HashMap::from_iter([(4, 48)])),
            (40, HashMap::from_iter([(3, 48), (2, 56)])),
            (56, HashMap::from_iter([(3, 24), (4, 56), (2, 24)])),
        ]);
        assert_eq!(index.transitions(), &expected);

        let allowed_tokens = index
            .allowed_tokens(&initial_state)
            .expect("No allowed tokens");
        let token_id = allowed_tokens.first().expect("No first tokens");

        let state = 48;
        assert_eq!(index.next_state(&initial_state, token_id), Some(state));
        assert!(index.is_final_state(&state));

        assert_eq!(index.next_state(&state, &eos_token_id), None);
        assert_eq!(index.next_state(&state, token_id), None);
    }

    #[test]
    fn index_from_regex_initital_in_allowed() {
        let regex = "`\\n(\\.\\n)?`\\n";
        let mut vocabulary = Vocabulary::new(104);
        for (token, token_id) in [("\n", 103), (".", 102), ("`", 101)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let allowed = index
            .allowed_tokens(&index.initial_state())
            .expect("No allowed tokens");
        assert!(allowed.contains(&101));
    }

    #[test]
    fn index_from_regex_multibyte() {
        let regex = "😇| [😈-😍][😇-😎]*";
        let mut vocabulary = Vocabulary::new(8);
        for (token, token_id) in [(" 😍", 5), ("blah", 0), ("😇", 2), ("😈a", 1), ("😍", 3)]
        {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        for (token, token_id) in [
            (vec![32, 240, 159, 152], 7),
            (vec![32, 240, 159, 152, 141], 6),
            (vec![240, 159, 152, 141], 4),
        ] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        assert_eq!(index.final_states(), &HashSet::from_iter([208, 128]));

        let expected = HashMap::from_iter([
            (
                208,
                HashMap::from_iter([(3, 208), (8, 208), (4, 208), (2, 208)]),
            ),
            (
                80,
                HashMap::from_iter([(2, 128), (7, 192), (5, 208), (6, 208)]),
            ),
            (128, HashMap::from_iter([(8, 128)])),
        ]);
        assert_eq!(index.transitions(), &expected);
    }

    #[test]
fn test_index_constructors() {
    let model_name = "unsloth/Llama-3.1-8B-Instruct";
    let regexes = get_bench_regexes();
    let vocab = Vocabulary::from_pretrained(model_name, None).unwrap();
    
    println!("> Benchmark Std Constructor vs Optim Constructor ({}) :", model_name);
    println!(
        "{:<45} | {:<20} | {:<20} | {:<15}",
        "Regex", "new()", "new_optimized()", "ratio"
    );
    println!(
        "{:<45} | {:<20} | {:<20} | {:<15} ",
        "-".repeat(45),
        "-".repeat(20),
        "-".repeat(20),
        "-".repeat(15),
       
    );

    for (name, regex) in &regexes{
        
        let schema: String;
        let regex_str = if name.contains("schema") {
            schema = json_schema::regex_from_str(regex, None).unwrap();
            schema.as_str()
        } else {
            regex
        };

        let start_new = Instant::now();
        let index_new = Index::new(regex_str, &vocab).expect("Failed to create Index with new");
        let duration_new = start_new.elapsed();

        let start_optimized = Instant::now();
        let index_optimized = Index::new_optimized(regex_str, &vocab).expect("Failed to create Index with new_optimized");
        let duration_optimized = start_optimized.elapsed();

        let time_new_ms = duration_new.as_secs_f64() * 1000.0;
        let time_optimized_ms = duration_optimized.as_secs_f64() * 1000.0;
        let ratio = if time_optimized_ms > 0.0 {
            time_new_ms / time_optimized_ms
        } else {
            f64::INFINITY
        };

        println!(
            "{:<45} | {:<20?} | {:<20?} | {:<15.2}x",
            name,
            duration_new,
            duration_optimized,
            ratio
        );
        let _ = std::io::stdout().flush();
    }

}



#[test]
fn test_index_text_generation_equivalence() {
    // Création d'un vocabulaire simple
    let mut vocab = Vocabulary::new(4); // EOS token ID = 4
    vocab.try_insert(b"a".to_vec(), 0 as u32).expect("Insert failed");      // Token "a"
    vocab.try_insert(b"b".to_vec(), 1 as u32).expect("Insert failed");      // Token "b"
    vocab.try_insert(b"ab".to_vec(), 2 as u32).expect("Insert failed");     // Token "ab"
    vocab.try_insert(b"abc".to_vec(), 3 as u32).expect("Insert failed");    // Token "abc"
    vocab.try_insert(b"ba".to_vec(), 5 as u32).expect("Insert failed");     // Token "ba"
    vocab.try_insert(b"bcd".to_vec(), 6 as u32).expect("Insert failed");    // Token "bcd"

    let regex = r"(bc|b|bcd){5}(a|ab)";

    // Construction des deux Index
    let index_new = Index::new(regex, &vocab).expect("Failed to create Index with new");
    let index_optimized = Index::new_optimized(regex, &vocab).expect("Failed to create Index with new_optimized");

    // Génération des textes
    let texts_new = index_new.generate_all_texts(&vocab);
    let texts_optimized = index_optimized.generate_all_texts(&vocab);

    // Comparaison des états et transitions
    println!("Index new: {} états, {} transitions", 
        index_new.transitions.len(), 
        index_new.count_total_transitions());
    println!("Index optimized: {} états, {} transitions", 
        index_optimized.transitions.len(), 
        index_optimized.count_total_transitions());
    
    // Comparaison des textes générés
    println!("Textes générés par new: {:?}", texts_new);
    println!("Textes générés par optimized: {:?}", texts_optimized);

    assert_eq!(texts_new, texts_optimized, "Les deux Index ne génèrent pas les mêmes textes !");
}

#[test]
fn test_email_reconstruction_with_gpt2_vocab() {
    // Charger le vocabulaire GPT-2
    let vocab = Vocabulary::from_pretrained("gpt2", None).unwrap();

    // Regex pour une adresse email (RFC 5322 simplifiée)
    let regex = r#"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])$"#;

    // Construire les deux Index
    let index_new = Index::new(regex, &vocab).expect("Failed to create Index with new");
    let index_optimized = Index::new_optimized(regex, &vocab).expect("Failed to create Index with new_optimized");

    // Tableau d'emails à tester
    let emails = vec![
        "machin.truc@truc-bidule.com",
        "user123@example.org",
        "john.doe+test@domain.co.uk",
        "a@b.com",
    ];

    // Tester chaque email
    for email in emails {
        println!("\nTesting email: {}", email);

        // Reconstruct avec index_new
        let (success_new, iterations_new) = reconstruct_email(&index_new, &vocab, email);
        println!("Index new: Success = {}, Iterations = {}", success_new, iterations_new);

        // Reconstruct avec index_optimized
        let (success_optimized, iterations_optimized) = reconstruct_email(&index_optimized, &vocab, email);
        println!("Index optimized: Success = {}, Iterations = {}", success_optimized, iterations_optimized);

        // Vérifier que les deux Index réussissent
        assert!(success_new, "Index new failed to reconstruct '{}'", email);
        assert!(success_optimized, "Index optimized failed to reconstruct '{}'", email);
    }
}

// Fonction pour reconstruire un email et compter les itérations
fn reconstruct_email(index: &Index, vocab: &Vocabulary, email: &str) -> (bool, usize) {
    let email_bytes = email.as_bytes();
    let mut current_state = index.initial_state;
    let mut position = 0;
    let mut iterations = 0;

    while position < email_bytes.len() {
        iterations += 1;

        // Récupérer les tokens autorisés depuis l'état actuel
        let empty_map: HashMap<TokenId, StateId> = HashMap::default();
        let allowed_tokens = index.transitions.get(&current_state).unwrap_or(&empty_map);

        // Trouver un token qui correspond à la partie suivante de l'email
        let mut found = false;
        for (&token_id, &next_state) in allowed_tokens {
            if token_id == index.eos_token_id {
                continue;
            }
            if let Some(token_bytes) = vocab.token_by_id(token_id) {
                if position + token_bytes.len() <= email_bytes.len() &&
                   &email_bytes[position..position + token_bytes.len()] == token_bytes.as_slice()
                {
                    current_state = next_state;
                    position += token_bytes.len();
                    found = true;
                    break;
                }
            }
        }

        // Si aucun token ne correspond, échec
        if !found {
            println!("Échec à la position {}: aucun token ne correspond", position);
            return (false, iterations);
        }
    }

    // Vérifier si l'état final est atteint
    let success = index.final_states.contains(&current_state);
    (success, iterations)
}

    #[test]
    fn test_minimal_index() {
        let mut vocab = Vocabulary::new(50256);
        vocab.try_insert(b"a".to_vec(), 0).unwrap();
        vocab.try_insert(b"b".to_vec(), 1).unwrap();
        vocab.try_insert(b"c".to_vec(), 2).unwrap();
        vocab.try_insert(b"d".to_vec(), 3).unwrap();
        vocab.try_insert(b"e".to_vec(), 4).unwrap();
        vocab.try_insert(b"f".to_vec(), 5).unwrap();
        vocab.try_insert(b"abcd".to_vec(), 6).unwrap();

        let regex = "abcdef$";

        let index = Index::new_optimized(regex, &vocab).unwrap();
        println!("États: {}", index.transitions.len());
        println!("Transitions: {}", index.transitions.values().map(|t| t.len()).sum::<usize>());
        println!("Transitions: {:?}", index.transitions);
        println!("Final states: {:?}", index.final_states);
    }

    fn get_bench_regexes() -> Vec<(&'static str, &'static str)> {
            vec![
                (
                    "email",
                    r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
                ),
                ("simple_phone", r"\+?[1-9][0-9]{7,14}"),
                (
                    "complex_phone",
                    r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
                ),
                ("permissive_any", r".{255}$"),
                ("permissive_words", r"[a-zA-Z]{100}"),
                (
                    "schema_simple",
                    r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#,
                ),
                (
                    "schema_simple_phone",
                    r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "complexe_phone": {"type": "string", "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"}}, "required": ["name", "age", "complexe_phone"]}"#,
                ),
                (
                    "schema_complexe",
                    r###"{
          "$schema": "http://json-schema.org/draft-04/schema#",
          "title": "Schema for a recording",
          "type": "object",
          "definitions": {
            "artist": {
              "type": "object",
              "properties": {
                "id": {"type": "number"},
                "name": {"type": "string"},
                "functions": {
                  "type": "array",
                  "items": {"type": "string"}
                }
              },
              "required": ["id", "name", "functions"]
            }
          },
          "properties": {
            "id": {"type": "number"},
            "work": {
              "type": "object",
              "properties": {
                "id": {"type": "number"},
                "name": {"type": "string"},
                "composer": {"$ref": "#/definitions/artist"}
              }
            },
            "recording_artists": {
              "type": "array",
              "items": {"$ref": "#/definitions/artist"}
            }
          },
          "required": ["id", "work", "recording_artists"]
        }"###,
                ),
            ]
    }
    
    
 
  
}
