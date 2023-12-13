

function handleFileSelect(event) {
    
    // return;
    // Obtém o arquivo selecionado
    const fileInput = event.target;
    const file = fileInput.files[0];

    // alert("me notaaa");
    // console.log("qualqquer coisa so pra testar");

    // Verifica se um arquivo foi selecionado
    if (file) {
        // Cria um objeto FormData para enviar o arquivo
        const formData = new FormData();
        formData.append('file', file);

        // Opções da solicitação
        const options = {
            method: 'POST',
            // method: 'GET',
            body: formData
        };

        // URL do servidor para onde você está enviando a solicitação POST
        const url = '/send_image';

        // Realiza a solicitação POST usando fetch
        // fetch(url, options)
        fetch(url, options)
            .then(response => response.json())
            .then(data => {

                console.log('Resposta do servidor:', data);

                // Converte a imagem para Base64
                const reader = new FileReader();
                reader.onloadend = function () {
                    // Cria uma lista contendo a imagem e outras informações
                    const listaConteudo = [reader.result, data["output"]];

                    // Armazena a lista no localStorage
                    localStorage.setItem('Image_0', JSON.stringify(listaConteudo));

                    // Redireciona para a nova tela
                    window.location.href = data["nova_url"];
                };


                reader.readAsDataURL(file);

            })
            .catch(error => {
                console.error('Erro:', error);
            });
        event.preventDefault();
    }
}



