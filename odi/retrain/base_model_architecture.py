from keras.layers import *
from keras.models import Model
from keras.regularizers import l2


def create_country_embedding_model(team_vector_len, opponent_vector_len, location_vector_len):
    team_input = Input((team_vector_len,), name="team_input")
    opponent_input = Input((opponent_vector_len,), name="opponent_input")
    location_input = Input((location_vector_len,), name="location_input")

    # team_output = Dropout(0.2)(team_input)
    team_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal', bias_regularizer=l2(0.01),
                        kernel_regularizer=l2(0.1), name="team_1")(team_input)
    team_output = Dropout(0.2)(team_output)

    # opponent_output = Dropout(0.2)(opponent_input)
    opponent_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                            bias_regularizer=l2(0.01), kernel_regularizer=l2(0.1), name="opp_1")(opponent_input)
    opponent_output = Dropout(0.2)(opponent_output)

    # location_output = Dropout(0.2)(location_input)
    location_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                            bias_regularizer=l2(0.01), kernel_regularizer=l2(0.1), name="loc_1")(location_input)
    location_output = Dropout(0.2)(location_output)

    concat_out = Concatenate()([team_output, opponent_output, location_output])
    runs_output = Dropout(0.2)(concat_out)
    runs_output = Dense(1, name="final_score", use_bias=True, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                        kernel_initializer='normal')(concat_out)

    team_model = Model(inputs=team_input, outputs=team_output)
    opponent_model = Model(inputs=opponent_input, outputs=opponent_output)
    location_model = Model(inputs=location_input, outputs=location_output)
    group_encode_model = Model(inputs=[team_input, opponent_input, location_input],
                               outputs=concat_out)

    runs_model = Model(inputs=[team_input, opponent_input, location_input],
                       outputs=runs_output)

    return team_model, opponent_model, location_model, group_encode_model, runs_model

def create_country_embedding_model_2nd(team_vector_len, opponent_vector_len, location_vector_len):
    team_input = Input((team_vector_len,), name="team_input")
    opponent_input = Input((opponent_vector_len,), name="opponent_input")
    location_input = Input((location_vector_len,), name="location_input")

    # team_output = Dropout(0.2)(team_input)
    team_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal', bias_regularizer=l2(0.01),
                        kernel_regularizer=l2(0.1), name="team_1")(team_input)
    team_output = Dropout(0.2)(team_output)

    # opponent_output = Dropout(0.2)(opponent_input)
    opponent_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                            bias_regularizer=l2(0.01), kernel_regularizer=l2(0.1), name="opp_1")(opponent_input)
    opponent_output = Dropout(0.2)(opponent_output)

    # location_output = Dropout(0.2)(location_input)
    location_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                            bias_regularizer=l2(0.01), kernel_regularizer=l2(0.1), name="loc_1")(location_input)
    location_output = Dropout(0.2)(location_output)


    concat_out = Concatenate()([team_output, opponent_output, location_output])
    #runs_output = Dropout(0.2)(concat_out)
    runs_output = Dense(1, activation='tanh', name="final_score", use_bias=True, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                        kernel_initializer='normal')(concat_out)

    team_model = Model(inputs=team_input, outputs=team_output)
    opponent_model = Model(inputs=opponent_input, outputs=opponent_output)
    location_model = Model(inputs=location_input, outputs=location_output)
    group_encode_model = Model(inputs=[team_input, opponent_input, location_input],
                               outputs=concat_out)

    runs_model = Model(inputs=[team_input, opponent_input, location_input],
                       outputs=runs_output)

    return team_model, opponent_model, location_model, group_encode_model, runs_model


def create_batsman_embedding_model(batsman_len, position_len, location_len, opposition_len):
    batsman_input = Input((batsman_len,), name="batsman_input")
    position_input = Input((position_len,), name="position_input")
    location_input = Input((location_len,), name="location_input")
    opposition_input = Input((opposition_len,), name="opposition_input")

    # team_output = Dropout(0.2)(team_input)
    batsman_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal', bias_regularizer=l2(0.01),
                           kernel_regularizer=l2(0.1), name="batsman_1")(batsman_input)
    batsman_output = Dropout(0.2)(batsman_output)

    # opponent_output = Dropout(0.2)(opponent_input)
    position_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                            bias_regularizer=l2(0.01), kernel_regularizer=l2(0.1), name="pos_1")(position_input)
    position_output = Dropout(0.2)(position_output)

    # location_output = Dropout(0.2)(location_input)
    location_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                            bias_regularizer=l2(0.01), kernel_regularizer=l2(0.1), name="loc_1")(location_input)
    location_output = Dropout(0.2)(location_output)

    opposition_output = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                              bias_regularizer=l2(0.01), kernel_regularizer=l2(0.1), name="opposition_1")(
        opposition_input)
    opposition_output = Dropout(0.2)(opposition_output)

    #     concat_out = Concatenate()([batsman_output, position_output,location_output,opposition_output])
    #     runs_output = Dropout(0.2)(concat_out)
    #     runs_output = Dense(1,name="final_score",use_bias=True, kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),kernel_initializer='normal')(concat_out)

    concat_out = Concatenate()([batsman_output, position_output, location_output, opposition_output])
    concat_out = Dropout(0.2)(concat_out)
    concat_out = Dense(10, name="concat_2", use_bias=True, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                       activation="relu", kernel_initializer='normal')(concat_out)
    runs_output = Dense(1, name="final_score", use_bias=True, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                        kernel_initializer='normal')(concat_out)

    batsman_model = Model(inputs=batsman_input, outputs=batsman_output)
    position_model = Model(inputs=position_input, outputs=position_output)
    location_model = Model(inputs=location_input, outputs=location_output)
    opposition_model = Model(inputs=opposition_input, outputs=opposition_output)
    group_encode_model = Model(inputs=[batsman_input, position_input, location_input, opposition_input],
                               outputs=concat_out)

    runs_model = Model(inputs=[batsman_input, position_input, location_input, opposition_input],
                       outputs=runs_output)

    return batsman_model, position_model, location_model, opposition_model, group_encode_model, runs_model


# Following models are used for reference #

def one_shot_multi_output_neural(first_innings_vector_length, second_innings_vector_length):
    first_innings_input = Input((first_innings_vector_length,), name="first_in")
    second_innings_input = Input((second_innings_vector_length,), name="second_in")



    first_innings_hidden = Dense(100, activation="relu", use_bias=True, kernel_initializer='normal', bias_regularizer=l2(0.01),
                        kernel_regularizer=l2(0.1), name="first_inn_hidden")(first_innings_input)
    first_innings_hidden_dropout = Dropout(0.2)(first_innings_hidden)
    first_innings_hidden_2 = Dense(10, name="first_inn_hidden_2", activation="relu",use_bias=True, kernel_regularizer=l2(0.01),
                                 bias_regularizer=l2(0.01),
                                 kernel_initializer='normal')(first_innings_hidden_dropout)
    first_innings_output = Dense(1, name="final_score", use_bias=True, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                        kernel_initializer='normal')(first_innings_hidden_2)

    second_innings_hidden = Dense(100, activation="relu", use_bias=True, kernel_initializer='normal',
                                  bias_regularizer=l2(0.01),
                                  kernel_regularizer=l2(0.1), name="second_inn_hidden")(second_innings_input)
    second_innings_hidden_dropout = Dropout(0.2)(second_innings_hidden)
    second_innings_hidden_2 = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                                  bias_regularizer=l2(0.01),
                                  kernel_regularizer=l2(0.1), name="second_inn_hidden_2")(second_innings_hidden_dropout)

    second_innings_hidden_3 = Concatenate()([first_innings_output, second_innings_hidden_2])
    # second_innings_hidden_4 = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
    #                                 bias_regularizer=l2(0.01),
    #                                 kernel_regularizer=l2(0.1), name="second_inn_hidden_4")(second_innings_hidden_3)

    second_innings_achieved_output = Dense(1, name="achieved_score", use_bias=True, kernel_regularizer=l2(0.01),
                                  bias_regularizer=l2(0.01),
                                  kernel_initializer='normal')(second_innings_hidden_3)
    second_innings_output = Dense(1, name="is_win", use_bias=True, kernel_regularizer=l2(0.01),
                                 bias_regularizer=l2(0.01),
                                 kernel_initializer='normal',activation="sigmoid")(second_innings_hidden_3)





    combined_model = Model(inputs=[first_innings_input, second_innings_input],
                       outputs=[first_innings_output,second_innings_achieved_output,second_innings_output])

    return combined_model

def one_shot_multi_output_neural_fs(first_innings_vector_length, second_innings_vector_length):
    first_innings_input = Input((first_innings_vector_length,), name="first_in")
    second_innings_input = Input((second_innings_vector_length,), name="second_in")



    first_innings_hidden = Dense(100, activation="relu", use_bias=True, kernel_initializer='normal', bias_regularizer=l2(0.01),
                        kernel_regularizer=l2(0.1), name="first_inn_hidden")(first_innings_input)
    first_innings_hidden_dropout = Dropout(0.2)(first_innings_hidden)
    first_innings_hidden_2 = Dense(10, name="first_inn_hidden_2", activation="relu",use_bias=True, kernel_regularizer=l2(0.01),
                                 bias_regularizer=l2(0.01),
                                 kernel_initializer='normal')(first_innings_hidden_dropout)
    first_innings_output = Dense(1, name="final_score", use_bias=True, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                        kernel_initializer='normal')(first_innings_hidden_2)

    second_innings_hidden = Dense(100, activation="relu", use_bias=True, kernel_initializer='normal',
                                  bias_regularizer=l2(0.01),
                                  kernel_regularizer=l2(0.1), name="second_inn_hidden")(second_innings_input)
    second_innings_hidden_dropout = Dropout(0.2)(second_innings_hidden)
    second_innings_hidden_2 = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                                  bias_regularizer=l2(0.01),
                                  kernel_regularizer=l2(0.1), name="second_inn_hidden_2")(second_innings_hidden_dropout)

    second_innings_hidden_3 = Concatenate()([first_innings_output, second_innings_hidden_2])
    # second_innings_hidden_4 = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
    #                                 bias_regularizer=l2(0.01),
    #                                 kernel_regularizer=l2(0.1), name="second_inn_hidden_4")(second_innings_hidden_3)

    second_innings_achieved_output = Dense(1, name="achieved_score", use_bias=True, kernel_regularizer=l2(0.01),
                                  bias_regularizer=l2(0.01),
                                  kernel_initializer='normal')(second_innings_hidden_3)
    second_innings_output = Dense(1, name="is_win", use_bias=True, kernel_regularizer=l2(0.01),
                                 bias_regularizer=l2(0.01),
                                 kernel_initializer='normal',activation="tanh")(second_innings_hidden_3)





    combined_model = Model(inputs=[first_innings_input, second_innings_input],
                       outputs=[first_innings_output,second_innings_achieved_output,second_innings_output])

    return combined_model



def create_sequential_model_with_inital_state(timesteps, embedding_lenght, inital_state_vector):
    """
    # team embedding can be used as inital state
    # total runs are final output
    # 11 timesteps for 11 batsman-each batsman embedding is used
    """
    sequence_input = Input((timesteps, embedding_lenght), name="sequence_input")
    initial_state = Input((inital_state_vector,), name="state_input")

    lstm_out = LSTM(inital_state_vector, activation='relu', return_sequences=False,
                    return_state=False, name='lstm_1')(sequence_input, initial_state=[initial_state, initial_state])
    runs_output = Dense(1, name='final_output')(lstm_out)

    runs_model = Model(inputs=[sequence_input, initial_state],
                       outputs=runs_output)

    return runs_model


def create_sequential_model(timesteps, embedding_lenght):
    """
    # team embedding can be used as inital state
    # total runs are final output
    # 11 timesteps for 11 batsman-each batsman embedding is used
    """
    sequence_input = Input((timesteps, embedding_lenght), name="sequence_input")

    lstm_out = LSTM(100, activation='relu', return_sequences=False,
                    return_state=False, name='lstm_1')(sequence_input)
    #     lstm_out = LSTM(40,activation='relu',return_sequences=False,
    #                     return_state=False,name='lstm_2')(lstm_out)
    #    lstm_out = Flatten()(lstm_out)

    runs_output = Dense(10, name='dense_1', activation='relu')(lstm_out)
    #    runs_output = Dense(5,name='dense_2',activation='relu')(runs_output)
    runs_output = Dense(1, name='final_output')(runs_output)

    runs_model = Model(inputs=[sequence_input],
                       outputs=runs_output)

    return runs_model


def create_seq2seq_model_with_inital_state(timesteps, embedding_lenght, inital_state_vector):
    """
    team embedding can be used as inital state
    total runs are final output
    11 timesteps for 11 batsman-each batsman embedding is used
    11 outputs from each sequence after timedistributed dens and flattening for runs of each plaer
    1 output as hidden state for total runs
    """
    sequence_input = Input((timesteps, embedding_lenght), name="sequence_input")
    initial_state = Input((inital_state_vector,), name="state_input")

    lstm_out, state_h, state_c = LSTM(inital_state_vector, activation='relu', return_sequences=True,
                                      return_state=True, name='lstm_1')(sequence_input,
                                                                        initial_state=[initial_state, initial_state])
    runs_output = TimeDistributed(Dense(1, name='ts_individual_output'))(lstm_out)

    runs_output = Flatten(name='individual_output')(runs_output)
    total_runs_output = Dense(1, name='total_output')(state_h)

    runs_model = Model(inputs=[sequence_input, initial_state],
                       outputs=[runs_output, total_runs_output])

    return runs_model


def create_dense_regression_model(input_len):

    team_input = Input((input_len,), name="team_input")

    # team_output = Dropout(0.2)(team_input)
    team_h1 = Dense(2*input_len, activation="relu", use_bias=True, kernel_initializer='normal',
                        bias_regularizer=l2(0.01),
                        kernel_regularizer=l2(0.1), name="team_h1")(team_input)
    team_h1_d = Dropout(0.2)(team_h1)

    team_h2 = Dense(2 * input_len, activation="relu", use_bias=True, kernel_initializer='normal',
                    bias_regularizer=l2(0.01),
                    kernel_regularizer=l2(0.1), name="team_h2")(team_h1_d)
    team_h2_d = Dropout(0.2)(team_h2)

    team_h3 = Dense(2 * input_len, activation="relu", use_bias=True, kernel_initializer='normal',
                    bias_regularizer=l2(0.01),
                    kernel_regularizer=l2(0.1), name="team_h3")(team_h2_d)
    team_h3_d = Dropout(0.2)(team_h3)

    team_h4 = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                    bias_regularizer=l2(0.01),
                    kernel_regularizer=l2(0.1), name="team_h4")(team_h3_d)
    team_h4_d = Dropout(0.2)(team_h4)

    runs_output = Dense(1, name="final_score", use_bias=True, kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01),
                        kernel_initializer='normal')(team_h4_d)

    runs_model = Model(inputs=[team_input],outputs=runs_output)

    return runs_model

def create_dense_classification_model(input_len):

    team_input = Input((input_len,), name="team_input")

    # team_output = Dropout(0.2)(team_input)
    team_h1 = Dense(2*input_len, activation="relu", use_bias=True, kernel_initializer='normal',
                        bias_regularizer=l2(0.01),
                        kernel_regularizer=l2(0.1), name="team_h1")(team_input)
    team_h1_d = Dropout(0.2)(team_h1)

    team_h2 = Dense(2 * input_len, activation="relu", use_bias=True, kernel_initializer='normal',
                    bias_regularizer=l2(0.01),
                    kernel_regularizer=l2(0.1), name="team_h2")(team_h1_d)
    team_h2_d = Dropout(0.2)(team_h2)

    team_h3 = Dense(2 * input_len, activation="relu", use_bias=True, kernel_initializer='normal',
                    bias_regularizer=l2(0.01),
                    kernel_regularizer=l2(0.1), name="team_h3")(team_h2_d)
    team_h3_d = Dropout(0.2)(team_h3)

    team_h4 = Dense(10, activation="relu", use_bias=True, kernel_initializer='normal',
                    bias_regularizer=l2(0.01),
                    kernel_regularizer=l2(0.1), name="team_h4")(team_h3_d)
    team_h4_d = Dropout(0.2)(team_h4)

    runs_output = Dense(1, name="result", activation="sigmoid", use_bias=True, kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01),
                        kernel_initializer='normal')(team_h4_d)

    runs_model = Model(inputs=[team_input],outputs=runs_output)

    return runs_model
