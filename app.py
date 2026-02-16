"""
Protein Function Classifier - Streamlit Web App
Predicts enzyme class (EC 1-7) from protein amino acid sequence.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Import our feature extraction module
from src.features import extract_all_features, get_feature_names

# EC class information
EC_CLASSES = {
    1: {"name": "Oxidoreductases", "description": "Catalyze oxidation-reduction reactions", "example": "Dehydrogenases, Oxidases"},
    2: {"name": "Transferases", "description": "Transfer functional groups between molecules", "example": "Kinases, Transaminases"},
    3: {"name": "Hydrolases", "description": "Catalyze hydrolysis reactions", "example": "Proteases, Lipases"},
    4: {"name": "Lyases", "description": "Break bonds without hydrolysis or oxidation", "example": "Decarboxylases, Aldolases"},
    5: {"name": "Isomerases", "description": "Catalyze structural rearrangements", "example": "Racemases, Mutases"},
    6: {"name": "Ligases", "description": "Join molecules using ATP", "example": "DNA Ligase, Synthetases"},
    7: {"name": "Translocases", "description": "Move molecules across membranes", "example": "ATP synthase, Ion pumps"}
}

# Sample sequences for demo
SAMPLE_SEQUENCES = {
    "Alcohol Dehydrogenase (EC 1)": "MSTAGKVIKCKAAVLWEEKKPFSIEEVEVAPPKAHEVRIKMVATGICRSDDHVVSGTLVTPLPVIAGHEAAGIVESIGEGVTTVRPGDKVIPLFTPQCGKCRVCKHPEGNFCLKNDLSMPRGTMQDGTSRFTCRGKPIHHFLGTSTFSQYTVVDEISVAKIDAASPLEKVCLIGCGFSTGYGSAVKVAKVTQGSTCAVFGLGGVGLSVIMGCKAAGAARIIGVDINKDKFAKAKEVGATECVNPQDYKKPIQEVLTEMSNGGVDFSFEVIGRLDTMVTALSCCQEAYGVSVIVGVPPDSQNLSMNPMLLLSGRTWKGAIFGGFKSKDSVPKLVADFMAKKFALDPLITHVLPFEKINEGFDLLRSGESIRTILTF",
    "Hexokinase (EC 2)": "MSLSQIERLDTLSATQQHLAQLGHTIVPLGFTFSYPASQNKINESLLQLKTNQQLGVIAALGTNGCGSGVRRRTLQRLLISRTPLDVELVEAADSIGLTRAVEQLHANGDLLLGEVGSGSVDAAGLESMVSHRLLRPVEVLPIVSPLDITINLDQNDLLLGLQRTFSPLENIDHAFESVAVVNDTVGTMMTCGYDDQHCERDVALLDMIMRMNRANPDTMLQKGEQAAFRHHLMRHLSLEETLDVLKGKLRKLLAQQVPQYSAIAGLHTGDMVRVLSPGNFAKFVDGLPEDTLCPLGWALNWCTGLGDGLVSWIKEKTREIAETLSAADDLRTSIAGCEFTGSGLHQCLPINQALAITDCEWGTPCQRQAGVMESLFGKGDAGLIQRYIDGLKKSNELV",
    "Trypsin (EC 3)": "MKTFIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN",
    "Carbonic Anhydrase (EC 4)": "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",
    "Triosephosphate Isomerase (EC 5)": "MRKFFVGGNWKMNGKKSLGELIHTLNGAKLSADTEVVCGAPSIYLDFARQKLDAKIGVAAQNCYKVPKGAFTGEISPAMIKDIGAAWVILGHSERRHVFGESDELIGQKVAHALAEGLGVIACIGEKLDEREAGITEKVVFEQTKAIADNVKDWSKVVLAYEPVWAIGTGKTATPQQAQEVHEKLRGWLKSNVSDAVAQSTRIIYGGSVTGATCKELASQPDVDGFLVGGASLKPEFVDIINAKQ",
    "DNA Ligase (EC 6)": "MTEQTLSLRQDLALLEKDKETLKQELPGVGQTIYVEGIVATTKPTGFVAGELKPYELLVEKDEKPTLILAEVPQGKGRKSAERLLKEYGFKVPNKLVLKVGADNVTCRSSLKDKAIIEPAAVLKKLSGCFVLAAGTRVGIVDPDTVNVLNRVLKERMQENGFKLIMVNRAMLSYVGTYLSAAGALLGNTDLASILAAYNPTLQAGLTDNAIARALGVDSLLTGQTVRDFIKEKGIKVITGSFDQPLLAQVKEALKRAGIDLAVSDVTLPAEPEGTPGQEQLAEVLKAHPDISILEVSAHFMQVDRSDPGLVEAAIAKIRALP"
}


@st.cache_resource
def load_model():
    """Load the trained model, scaler, and label encoder."""
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'
    encoder_path = 'models/label_encoder.pkl'
    
    if not os.path.exists(model_path):
        st.error("Model not found! Please train the model first.")
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    
    return model, scaler, encoder


def validate_sequence(sequence):
    """Validate protein sequence."""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    sequence = sequence.upper().replace(' ', '').replace('\n', '')
    
    invalid_chars = set(sequence) - valid_aa
    if invalid_chars:
        return None, f"Invalid characters found: {invalid_chars}"
    
    if len(sequence) < 50:
        return None, "Sequence too short (minimum 50 amino acids)"
    
    if len(sequence) > 2000:
        return None, "Sequence too long (maximum 2000 amino acids)"
    
    return sequence, None


def predict_function(sequence, model, scaler, encoder):
    """Predict protein function from sequence."""
    # Extract features
    features = extract_all_features(sequence)
    features = np.array(features).reshape(1, -1)
    
    # Scale features
    if scaler is not None:
        features = scaler.transform(features)
    
    # Predict
    pred_encoded = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Decode prediction back to original EC class
    if encoder is not None:
        pred_class = encoder.inverse_transform([pred_encoded])[0]
    else:
        pred_class = pred_encoded + 1  # Fallback
    
    return pred_class, probabilities


def main():
    # Page config
    st.set_page_config(
        page_title="Protein Function Classifier",
        page_icon="🧬",
        layout="wide"
    )
    
    # Header
    st.title("🧬 Protein Function Classifier")
    st.markdown("""
    Predict the **enzyme class (EC number)** of a protein from its amino acid sequence.
    
    This model classifies proteins into one of **7 major enzyme classes** using machine learning
    trained on UniProt data.
    """)
    
    # Load model
    model, scaler, encoder = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Model:** XGBoost Classifier
        
        **Features:** 437 features including:
        - Amino acid composition (20)
        - Dipeptide composition (400)
        - Physicochemical properties (10)
        - Secondary structure propensity (4)
        - Sequence complexity (3)
        
        **Training Data:** ~5,300 sequences from UniProt
        """)
        
        st.header("📊 EC Classes")
        for ec, info in EC_CLASSES.items():
            with st.expander(f"EC {ec}: {info['name']}"):
                st.write(info['description'])
                st.caption(f"Examples: {info['example']}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Enter Protein Sequence")
        
        # Sample sequence selector
        sample_choice = st.selectbox(
            "Or try a sample sequence:",
            ["-- Select a sample --"] + list(SAMPLE_SEQUENCES.keys())
        )
        
        # Text input
        if sample_choice != "-- Select a sample --":
            default_seq = SAMPLE_SEQUENCES[sample_choice]
        else:
            default_seq = ""
        
        sequence_input = st.text_area(
            "Paste amino acid sequence (single letter codes):",
            value=default_seq,
            height=200,
            placeholder="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH..."
        )
        
        predict_button = st.button("🔬 Predict Function", type="primary", use_container_width=True)
    
    with col2:
        st.header("Sequence Info")
        if sequence_input:
            clean_seq = sequence_input.upper().replace(' ', '').replace('\n', '')
            st.metric("Length", f"{len(clean_seq)} aa")
            
            # Quick composition
            if len(clean_seq) > 0:
                charged = sum(1 for aa in clean_seq if aa in 'DEKRH') / len(clean_seq)
                hydrophobic = sum(1 for aa in clean_seq if aa in 'AILMFVWY') / len(clean_seq)
                st.metric("Charged residues", f"{charged:.1%}")
                st.metric("Hydrophobic residues", f"{hydrophobic:.1%}")
    
    # Prediction
    if predict_button and sequence_input:
        # Validate
        sequence, error = validate_sequence(sequence_input)
        
        if error:
            st.error(f"❌ {error}")
        else:
            with st.spinner("Analyzing sequence..."):
                pred_class, probabilities = predict_function(sequence, model, scaler, encoder)
            
            # Results
            st.header("🎯 Prediction Results")
            
            # Main prediction
            pred_info = EC_CLASSES[pred_class]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.success(f"## EC {pred_class}")
                st.subheader(pred_info['name'])
                st.write(pred_info['description'])
                st.caption(f"Examples: {pred_info['example']}")
            
            with col2:
                st.subheader("Confidence Scores")
                
                # Create probability dataframe
                if encoder is not None:
                    classes = encoder.inverse_transform(range(len(probabilities)))
                else:
                    classes = range(1, 8)
                
                prob_df = pd.DataFrame({
                    'EC Class': [f"EC {c}: {EC_CLASSES[c]['name']}" for c in classes],
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                # Display as horizontal bars
                for _, row in prob_df.iterrows():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(row['Probability'], text=row['EC Class'])
                    with col_b:
                        st.write(f"{row['Probability']:.1%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Built with Streamlit | Model trained on UniProt enzyme data<br>
        <a href='https://github.com/Manju-Selvakumaran/protein-function-classifier'>GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()